// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL

#include "video-decoder.h"

#include <jni.h>

#include <android/native_window_jni.h>
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaFormat.h>

#include <string.h>
#include <stdlib.h>

#define INPUT_BUFFER_TIMEOUT_MS 10
#define OUTPUT_BUFFER_TIMEOUT_US 50000
#define MAX_RECOVERY_ATTEMPTS 3

static void *android_chiaki_video_decoder_output_thread_func(void *user);
static bool queue_sample_locked(AndroidChiakiVideoDecoder *decoder,
                                const uint8_t *buf, size_t buf_size,
                                bool *fatal);

ChiakiErrorCode
android_chiaki_video_decoder_init(AndroidChiakiVideoDecoder *decoder,
                                  ChiakiLog *log, int32_t target_width,
                                  int32_t target_height, ChiakiCodec codec) {
  decoder->log = log;
  decoder->codec = NULL;
  decoder->timestamp_cur = 0;
  decoder->target_width = target_width;
  decoder->target_height = target_height;
  decoder->target_codec = codec;
  decoder->shutdown_output = false;
  decoder->output_thread_started = false;
  decoder->output_failed = false;
  decoder->recovery_attempts = 0;
  decoder->successful_outputs = 0;
  decoder->window = NULL;
  decoder->codec_header = NULL;
  decoder->codec_header_size = 0;
  ChiakiErrorCode err = chiaki_mutex_init(&decoder->lifecycle_mutex, false);
  if (err != CHIAKI_ERR_SUCCESS)
    return err;
  err = chiaki_mutex_init(&decoder->codec_mutex, false);
  if (err != CHIAKI_ERR_SUCCESS)
    chiaki_mutex_fini(&decoder->lifecycle_mutex);
  return err;
}

// Caller holds lifecycle_mutex. Never hold codec_mutex while joining the
// output thread: that thread takes codec_mutex when the codec is stopped.
static void kill_decoder(AndroidChiakiVideoDecoder *decoder) {
  chiaki_mutex_lock(&decoder->codec_mutex);
  if (!decoder->codec) {
    chiaki_mutex_unlock(&decoder->codec_mutex);
    return;
  }
  decoder->shutdown_output = true;
  AMediaCodec_stop(decoder->codec);
  chiaki_mutex_unlock(&decoder->codec_mutex);

  if (decoder->output_thread_started)
    chiaki_thread_join(&decoder->output_thread, NULL);

  chiaki_mutex_lock(&decoder->codec_mutex);
  AMediaCodec_delete(decoder->codec);
  decoder->codec = NULL;
  if (decoder->window)
    ANativeWindow_release(decoder->window);
  decoder->window = NULL;
  decoder->shutdown_output = false;
  decoder->output_thread_started = false;
  decoder->output_failed = false;
  chiaki_mutex_unlock(&decoder->codec_mutex);
}

void android_chiaki_video_decoder_fini(AndroidChiakiVideoDecoder *decoder) {
  chiaki_mutex_lock(&decoder->lifecycle_mutex);
  kill_decoder(decoder);
  free(decoder->codec_header);
  decoder->codec_header = NULL;
  decoder->codec_header_size = 0;
  chiaki_mutex_unlock(&decoder->lifecycle_mutex);
  chiaki_mutex_fini(&decoder->codec_mutex);
  chiaki_mutex_fini(&decoder->lifecycle_mutex);
}

// Profile headers contain an Annex B SPS. Keep the most recent header so a
// recreated codec can decode the IDR requested after a rejected frame.
static bool is_codec_header(AndroidChiakiVideoDecoder *decoder,
                            const uint8_t *buf, size_t size) {
  for (size_t i = 0; i + 4 < size; i++) {
    size_t nal = 0;
    if (buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1)
      nal = i + 3;
    else if (i + 5 < size && buf[i] == 0 && buf[i + 1] == 0 &&
             buf[i + 2] == 0 && buf[i + 3] == 1)
      nal = i + 4;
    if (nal && (chiaki_codec_is_h265(decoder->target_codec)
                    ? ((buf[nal] >> 1) & 0x3f) == 33
                    : (buf[nal] & 0x1f) == 7))
      return true;
  }
  return false;
}

// Caller holds codec_mutex.
static bool queue_sample_locked(AndroidChiakiVideoDecoder *decoder,
                                const uint8_t *buf, size_t buf_size,
                                bool *fatal) {
  *fatal = false;
  while (buf_size > 0) {
    ssize_t index = AMediaCodec_dequeueInputBuffer(
        decoder->codec, INPUT_BUFFER_TIMEOUT_MS * 1000);
    if (index < 0) {
      CHIAKI_LOGW(decoder->log, "Video codec input buffer unavailable (%zd)", index);
      *fatal = index != AMEDIACODEC_INFO_TRY_AGAIN_LATER;
      return false;
    }
    size_t capacity = 0;
    uint8_t *input = AMediaCodec_getInputBuffer(decoder->codec, (size_t)index,
                                                &capacity);
    if (!input || !capacity) {
      CHIAKI_LOGE(decoder->log, "Video codec returned an empty input buffer");
      *fatal = true;
      return false;
    }
    size_t chunk = buf_size < capacity ? buf_size : capacity;
    memcpy(input, buf, chunk);
    media_status_t status = AMediaCodec_queueInputBuffer(
        decoder->codec, (size_t)index, 0, chunk, decoder->timestamp_cur++, 0);
    if (status != AMEDIA_OK) {
      CHIAKI_LOGE(decoder->log, "AMediaCodec_queueInputBuffer() failed: %d",
                  (int)status);
      *fatal = true;
      return false;
    }
    buf += chunk;
    buf_size -= chunk;
  }
  return true;
}

// Caller holds lifecycle_mutex. Takes ownership of window even on failure.
static bool start_decoder(AndroidChiakiVideoDecoder *decoder,
                          ANativeWindow *window) {
  if (!window)
    return false;
  chiaki_mutex_lock(&decoder->codec_mutex);
  decoder->window = window;

  const char *mime =
      chiaki_codec_is_h265(decoder->target_codec) ? "video/hevc" : "video/avc";
  CHIAKI_LOGI(decoder->log, "Initializing decoder with mime %s", mime);

  decoder->codec = AMediaCodec_createDecoderByType(mime);
  if (!decoder->codec) {
    CHIAKI_LOGE(decoder->log, "Failed to create AMediaCodec for mime type %s",
                mime);
    goto error_surface;
  }

  AMediaFormat *format = AMediaFormat_new();
  if (!format) {
    CHIAKI_LOGE(decoder->log, "Failed to allocate video decoder format");
    goto error_codec;
  }
  AMediaFormat_setString(format, AMEDIAFORMAT_KEY_MIME, mime);
  AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_WIDTH, decoder->target_width);
  AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_HEIGHT,
                        decoder->target_height);

  media_status_t r =
      AMediaCodec_configure(decoder->codec, format, decoder->window, NULL, 0);
  if (r != AMEDIA_OK) {
    CHIAKI_LOGE(decoder->log, "AMediaCodec_configure() failed: %d", (int)r);
    AMediaFormat_delete(format);
    goto error_codec;
  }

  r = AMediaCodec_start(decoder->codec);
  AMediaFormat_delete(format);
  if (r != AMEDIA_OK) {
    CHIAKI_LOGE(decoder->log, "AMediaCodec_start() failed: %d", (int)r);
    goto error_codec;
  }

  ChiakiErrorCode err = chiaki_thread_create(
      &decoder->output_thread, android_chiaki_video_decoder_output_thread_func,
      decoder);
  if (err != CHIAKI_ERR_SUCCESS) {
    CHIAKI_LOGE(decoder->log, "Failed to create output thread for AMediaCodec");
    AMediaCodec_stop(decoder->codec);
    goto error_codec;
  }
  decoder->output_thread_started = true;
  decoder->output_failed = false;
  bool header_fatal = false;
  if (decoder->codec_header_size &&
      !queue_sample_locked(decoder, decoder->codec_header,
                           decoder->codec_header_size, &header_fatal))
    decoder->output_failed = true;

  chiaki_mutex_unlock(&decoder->codec_mutex);
  return true;

error_codec:
  AMediaCodec_delete(decoder->codec);
  decoder->codec = NULL;

error_surface:
  ANativeWindow_release(decoder->window);
  decoder->window = NULL;

  chiaki_mutex_unlock(&decoder->codec_mutex);
  return false;
}

void android_chiaki_video_decoder_set_surface(
    AndroidChiakiVideoDecoder *decoder, JNIEnv *env, jobject surface) {
  chiaki_mutex_lock(&decoder->lifecycle_mutex);
  if (!surface) {
    kill_decoder(decoder);
    chiaki_mutex_lock(&decoder->codec_mutex);
    decoder->recovery_attempts = 0;
    decoder->successful_outputs = 0;
    chiaki_mutex_unlock(&decoder->codec_mutex);
    CHIAKI_LOGI(decoder->log, "Decoder shut down after surface was removed");
    chiaki_mutex_unlock(&decoder->lifecycle_mutex);
    return;
  }

  ANativeWindow *new_window = ANativeWindow_fromSurface(env, surface);
  if (!new_window) {
    CHIAKI_LOGE(decoder->log, "Could not create native video window");
    chiaki_mutex_unlock(&decoder->lifecycle_mutex);
    return;
  }

  chiaki_mutex_lock(&decoder->codec_mutex);
  if (decoder->codec && !decoder->output_failed) {
    media_status_t status = AMediaCodec_setOutputSurface(decoder->codec, new_window);
    if (status == AMEDIA_OK) {
      ANativeWindow *old_window = decoder->window;
      decoder->window = new_window;
      chiaki_mutex_unlock(&decoder->codec_mutex);
      if (old_window)
        ANativeWindow_release(old_window);
      chiaki_mutex_unlock(&decoder->lifecycle_mutex);
      return;
    }
    CHIAKI_LOGW(decoder->log, "Surface swap failed (%d); recreating decoder",
                (int)status);
  }
  chiaki_mutex_unlock(&decoder->codec_mutex);
  kill_decoder(decoder);
  chiaki_mutex_lock(&decoder->codec_mutex);
  decoder->recovery_attempts = 0;
  decoder->successful_outputs = 0;
  chiaki_mutex_unlock(&decoder->codec_mutex);
  start_decoder(decoder, new_window);
  chiaki_mutex_unlock(&decoder->lifecycle_mutex);
}

static bool recreate_decoder(AndroidChiakiVideoDecoder *decoder) {
  chiaki_mutex_lock(&decoder->lifecycle_mutex);
  chiaki_mutex_lock(&decoder->codec_mutex);
  ANativeWindow *window = decoder->window;
  if (window)
    ANativeWindow_acquire(window);
  chiaki_mutex_unlock(&decoder->codec_mutex);
  if (!window) {
    chiaki_mutex_unlock(&decoder->lifecycle_mutex);
    return false;
  }
  kill_decoder(decoder);
  CHIAKI_LOGW(decoder->log, "Recreating failed video codec");
  bool started = start_decoder(decoder, window);
  chiaki_mutex_unlock(&decoder->lifecycle_mutex);
  return started;
}

bool android_chiaki_video_decoder_video_sample(uint8_t *buf, size_t buf_size,
                                               int32_t frames_lost,
                                               bool frame_recovered,
                                               void *user) {
  (void)frames_lost;     // unused
  (void)frame_recovered; // unused
  AndroidChiakiVideoDecoder *decoder = user;
  chiaki_mutex_lock(&decoder->codec_mutex);

  bool header = buf_size <= 8192 && is_codec_header(decoder, buf, buf_size);
  if (header) {
    uint8_t *copy = malloc(buf_size);
    if (copy) {
      memcpy(copy, buf, buf_size);
      free(decoder->codec_header);
      decoder->codec_header = copy;
      decoder->codec_header_size = buf_size;
    }
  }

  // A profile header may arrive before Android has supplied a surface.
  if (!decoder->codec || decoder->shutdown_output) {
    chiaki_mutex_unlock(&decoder->codec_mutex);
    return header;
  }
  bool fatal = decoder->output_failed;
  bool queued = !fatal &&
                queue_sample_locked(decoder, buf, buf_size, &fatal);
  bool recover = !queued && fatal &&
                 decoder->recovery_attempts < MAX_RECOVERY_ATTEMPTS;
  if (recover)
    decoder->recovery_attempts++;
  if (recover)
    decoder->successful_outputs = 0;
  chiaki_mutex_unlock(&decoder->codec_mutex);
  if (recover)
    recreate_decoder(decoder);
  // A rejected frame must not be recorded as a usable reference frame.
  return queued;
}

static void *android_chiaki_video_decoder_output_thread_func(void *user) {
  AndroidChiakiVideoDecoder *decoder = user;

  while (1) {
    AMediaCodecBufferInfo info;
    ssize_t status = AMediaCodec_dequeueOutputBuffer(
        decoder->codec, &info, OUTPUT_BUFFER_TIMEOUT_US);
    if (status >= 0) {
      media_status_t released = AMediaCodec_releaseOutputBuffer(
          decoder->codec, (size_t)status, info.size != 0);
      if (released != AMEDIA_OK) {
        CHIAKI_LOGE(decoder->log, "Video output release failed: %d", (int)released);
        chiaki_mutex_lock(&decoder->codec_mutex);
        decoder->output_failed = true;
        chiaki_mutex_unlock(&decoder->codec_mutex);
        break;
      }
      if (info.size != 0) {
        chiaki_mutex_lock(&decoder->codec_mutex);
        if (++decoder->successful_outputs >= 120) {
          decoder->recovery_attempts = 0;
          decoder->successful_outputs = 0;
        }
        chiaki_mutex_unlock(&decoder->codec_mutex);
      }
      if (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) {
        CHIAKI_LOGI(decoder->log, "AMediaCodec reported EOS");
        chiaki_mutex_lock(&decoder->codec_mutex);
        if (!decoder->shutdown_output)
          decoder->output_failed = true;
        chiaki_mutex_unlock(&decoder->codec_mutex);
        break;
      }
    } else {
      chiaki_mutex_lock(&decoder->codec_mutex);
      bool shutdown = decoder->shutdown_output;
      if (!shutdown && status != AMEDIACODEC_INFO_TRY_AGAIN_LATER &&
          status != AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED &&
          status != AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {
        CHIAKI_LOGE(decoder->log, "Video output dequeue failed: %zd", status);
        decoder->output_failed = true;
      }
      bool failed = decoder->output_failed;
      chiaki_mutex_unlock(&decoder->codec_mutex);
      if (shutdown || failed) {
        CHIAKI_LOGI(decoder->log, "Video Decoder Output Thread detected "
                                  "shutdown after reported error");
        break;
      }
    }
  }

  CHIAKI_LOGI(decoder->log, "Video Decoder Output Thread exiting");

  return NULL;
}
