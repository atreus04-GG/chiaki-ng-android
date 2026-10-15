// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL

package com.metallic.chiaki.stream

import android.graphics.SurfaceTexture
import android.opengl.GLES11Ext
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.util.Log
import android.view.Surface
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.random.Random

/**
 * GLSurfaceView.Renderer that applies debanding shader to video frames.
 * Uses SurfaceTexture to receive frames from MediaCodec and renders them
 * with a GLSL debanding effect.
 */
class DebandRenderer(
    private val sourceWidth: Int,
    private val sourceHeight: Int,
    private val debandingEnabled: Boolean,
    private val fsrEnabled: Boolean,
    private val rcasEnabled: Boolean,
    private val requestRender: () -> Unit,
    private val onRendererFailure: () -> Unit,
    private val onSurfaceReady: (Surface) -> Unit
) : GLSurfaceView.Renderer, SurfaceTexture.OnFrameAvailableListener {

    companion object {
        private const val TAG = "DebandRenderer"

        // 1. Copy pass (OES -> FBO)
        private const val COPY_VERTEX_SHADER = """
            #version 300 es
            in vec4 a_Position;
            in vec2 a_TexCoord;
            out highp vec2 v_TexCoord;
            uniform mat4 u_STMatrix;
            void main() {
                gl_Position = a_Position;
                v_TexCoord = (u_STMatrix * vec4(a_TexCoord, 0.0, 1.0)).xy;
            }
        """

        private const val COPY_FRAGMENT_SHADER = """
            #version 300 es
            #extension GL_OES_EGL_image_external_essl3 : require
            precision highp float;
            in highp vec2 v_TexCoord;
            out vec4 outColor;
            uniform samplerExternalOES u_Texture;
            void main() {
                outColor = texture(u_Texture, v_TexCoord);
            }
        """

        // 2. Effects pass (FBO -> Screen)
        private const val EFFECT_VERTEX_SHADER = """
            #version 300 es
            in vec4 a_Position;
            in vec2 a_TexCoord;
            out highp vec2 v_TexCoord;
            void main() {
                gl_Position = a_Position;
                // FBO texture is usually upside down relative to OES coordinates
                v_TexCoord = a_TexCoord;
            }
        """

        private const val EFFECT_FRAGMENT_SHADER = """
            #version 300 es
            precision highp float;

            in highp vec2 v_TexCoord;
            out vec4 outColor;

            uniform sampler2D u_Texture;
            uniform highp float u_Time;
            uniform highp vec2 u_ScreenSize;
            uniform highp float u_Sharpness;
            uniform bool u_Deband;

            // Stochastic Debanding v2 Parameters
            const float DEBAND_THRESHOLD = 0.02;   // Sensitivity to banding steps (lower = safer for textures)
            const int NUM_SAMPLES = 24;            // More samples = smoother gradients
            const float MAX_RADIUS = 24.0;         // Effective sampling radius
            const float GRAIN_STRENGTH = 0.003;    // Slightly reduced grain

            // Interleaved Gradient Noise
            highp float ign(vec2 v) {
                v = floor(v * u_ScreenSize);
                return fract(52.9829189 * fract(dot(v, vec2(0.06711056, 0.00583715))));
            }

            // High quality fast PRNG
            highp float rand(vec2 co, float seed) {
                return fract(sin(dot(co + seed, vec2(12.9898, 78.233))) * 43758.5453);
            }

            void main() {
                highp vec2 texelSize = 1.0 / u_ScreenSize;
                vec3 original = texture(u_Texture, v_TexCoord).rgb;
                
                // ═══════════════════════════════════════════════════════════════
                // STOCHASTIC DEBANDING (libplacebo-inspired)
                // ═══════════════════════════════════════════════════════════════
                
                vec3 sum = original;
                float totalW = 1.0;
                
                // Use IGN + Time for jittered sampling
                float noise = ign(v_TexCoord);
                float timeSeed = fract(u_Time * 0.1);
                
                for (int i = 0; i < NUM_SAMPLES; i++) {
                    if (!u_Deband) break;
                    // Generate pseudo-random angle and radius
                    float fi = float(i);
                    float angle = (fi + noise) * 2.3999632; // Golden angle for distribution
                    float r = sqrt((fi + 0.5) / float(NUM_SAMPLES)) * MAX_RADIUS;
                    
                    vec2 offset = vec2(cos(angle), sin(angle)) * r * texelSize;
                    vec3 s = texture(u_Texture, clamp(v_TexCoord + offset, 0.0, 1.0)).rgb;
                    
                    // Difference check: only average pixels that could be part of the same gradient
                    float diff = max(max(abs(original.r - s.r), abs(original.g - s.g)), abs(original.b - s.b));
                    
                    // Soft threshold: skip edges, keep gradients. 
                    // Lower threshold means we only blend very similar colors.
                    float w = 1.0 - smoothstep(0.0, DEBAND_THRESHOLD, diff);
                    sum += s * w;
                    totalW += w;
                }
                
                vec3 debanded = sum / totalW;
                vec3 color = debanded;

                // ═══════════════════════════════════════════════════════════════
                // RCAS (Robust Contrast Adaptive Sharpening)
                // Adapted from AMD FSR 1; see LICENSES/AMD-FSR1-MIT.txt.
                // ═══════════════════════════════════════════════════════════════
                
                // Sample neighbors from the original texture for better edge detection
                vec3 b = texture(u_Texture, v_TexCoord + vec2(0.0, -texelSize.y)).rgb;
                vec3 d = texture(u_Texture, v_TexCoord + vec2(-texelSize.x, 0.0)).rgb;
                vec3 f = texture(u_Texture, v_TexCoord + vec2(texelSize.x, 0.0)).rgb;
                vec3 h = texture(u_Texture, v_TexCoord + vec2(0.0, texelSize.y)).rgb;
                
                if (u_Sharpness > 0.0) {
                    vec3 e = color;
                    vec3 minimum = min(min(b, d), min(f, h));
                    vec3 maximum = max(max(b, d), max(f, h));
                    vec3 hitMin = min(minimum, e) / max(4.0 * maximum, vec3(1e-6));
                    vec3 hitMax = (1.0 - max(maximum, e)) / (4.0 * minimum - 4.0);
                    vec3 lobePerChannel = max(-hitMin, hitMax);
                    float lobe = max(-0.1875,
                        min(max(lobePerChannel.r, max(lobePerChannel.g, lobePerChannel.b)), 0.0));
                    lobe *= u_Sharpness;
                    color = clamp((lobe * (b + d + f + h) + e) / (1.0 + 4.0 * lobe), 0.0, 1.0);
                }

                // ═══════════════════════════════════════════════════════════════
                // FINAL DITHER
                // ═══════════════════════════════════════════════════════════════
                if (u_Deband) {
                    float dither = (ign(v_TexCoord + timeSeed) - 0.5) * 0.005;
                    color += vec3(dither + GRAIN_STRENGTH * (rand(v_TexCoord, timeSeed) - 0.5));
                }

                outColor = vec4(color, 1.0);
            }
        """

        // Fullscreen quad vertices (position + texcoord)
        private val QUAD_VERTICES = floatArrayOf(
            // X,    Y,    U,    V
            -1.0f, -1.0f, 0.0f, 0.0f,
             1.0f, -1.0f, 1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f, 1.0f,
             1.0f,  1.0f, 1.0f, 1.0f
        )
        private const val COORDS_PER_VERTEX = 4
        private const val VERTEX_STRIDE = COORDS_PER_VERTEX * 4 // 4 bytes per float
    }

    // OpenGL programs
    private var copyProgram = 0
    private var effectProgram = 0
    private var fsrProgram = 0
    
    // Textures and FBO
    private var oesTextureId = 0
    private var fboTextureId = 0
    private var fboId = 0
    private var fsrTextureId = 0
    private var fsrFboId = 0
    private var fsrActive = false
    
    private var vertexBuffer: FloatBuffer? = null

    // Pass 1 (Copy) locations
    private var uCopySTMatrixLoc = 0
    private var uCopyTextureLoc = 0
    private var aCopyPosLoc = 0
    private var aCopyTexLoc = 0
    
    // Pass 2 (Effect) locations
    private var uEffTextureLoc = 0
    private var uEffTimeLoc = 0
    private var uEffScreenSizeLoc = 0
    private var uEffSharpnessLoc = 0
    private var uEffDebandLoc = 0
    private var aEffPosLoc = 0
    private var aEffTexLoc = 0
    private var uFsrTextureLoc = 0
    private var uFsrOutputSizeLoc = 0
    private var aFsrPosLoc = 0
    private var aFsrTexLoc = 0

    // SurfaceTexture and Surface for MediaCodec output
    private var surfaceTexture: SurfaceTexture? = null
    private var surface: Surface? = null

    // Texture transform matrix
    private val stMatrix = FloatArray(16)

    var sharpness = 0f

    // Screen dimensions
    private var screenWidth = 1
    private var screenHeight = 1

    @Volatile
    private var frameAvailable = false
    private var frameCount = 0f
    private var rendererFailed = false

    init {
        android.opengl.Matrix.setIdentityM(stMatrix, 0)
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        // 1. Create copy program
        copyProgram = createProgram(COPY_VERTEX_SHADER, COPY_FRAGMENT_SHADER)
        effectProgram = createProgram(EFFECT_VERTEX_SHADER, EFFECT_FRAGMENT_SHADER)
        if (copyProgram == 0 || effectProgram == 0) {
            failRenderer()
            return
        }
        aCopyPosLoc = GLES30.glGetAttribLocation(copyProgram, "a_Position")
        aCopyTexLoc = GLES30.glGetAttribLocation(copyProgram, "a_TexCoord")
        uCopySTMatrixLoc = GLES30.glGetUniformLocation(copyProgram, "u_STMatrix")
        uCopyTextureLoc = GLES30.glGetUniformLocation(copyProgram, "u_Texture")

        // 2. Create effect program
        aEffPosLoc = GLES30.glGetAttribLocation(effectProgram, "a_Position")
        aEffTexLoc = GLES30.glGetAttribLocation(effectProgram, "a_TexCoord")
        uEffTextureLoc = GLES30.glGetUniformLocation(effectProgram, "u_Texture")
        uEffTimeLoc = GLES30.glGetUniformLocation(effectProgram, "u_Time")
        uEffScreenSizeLoc = GLES30.glGetUniformLocation(effectProgram, "u_ScreenSize")
        uEffSharpnessLoc = GLES30.glGetUniformLocation(effectProgram, "u_Sharpness")
        uEffDebandLoc = GLES30.glGetUniformLocation(effectProgram, "u_Deband")

        if (fsrEnabled) {
            fsrProgram = createProgram(EFFECT_VERTEX_SHADER, Fsr1Easu.FRAGMENT_SHADER)
            if (fsrProgram != 0) {
                aFsrPosLoc = GLES30.glGetAttribLocation(fsrProgram, "a_Position")
                aFsrTexLoc = GLES30.glGetAttribLocation(fsrProgram, "a_TexCoord")
                uFsrTextureLoc = GLES30.glGetUniformLocation(fsrProgram, "u_Texture")
                uFsrOutputSizeLoc = GLES30.glGetUniformLocation(fsrProgram, "u_OutputSize")
            } else {
                Log.w(TAG, "FSR shader unavailable; using linear scaling")
            }
        }

        // 3. Create OES texture
        val textures = IntArray(3)
        GLES30.glGenTextures(3, textures, 0)
        oesTextureId = textures[0]
        fboTextureId = textures[1]
        fsrTextureId = textures[2]

        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTextureId)
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        
        // 4. Create FBO
        val fbos = IntArray(2)
        GLES30.glGenFramebuffers(2, fbos, 0)
        fboId = fbos[0]
        fsrFboId = fbos[1]

        // 5. Create SurfaceTexture
        val oldSurface = surface
        val oldTexture = surfaceTexture
        surfaceTexture = SurfaceTexture(oesTextureId).also {
            it.setOnFrameAvailableListener(this)
            it.setDefaultBufferSize(sourceWidth, sourceHeight)
            surface = Surface(it)
            onSurfaceReady(surface!!)
        }
        oldSurface?.release()
        oldTexture?.release()

        // Create vertex buffer
        vertexBuffer = ByteBuffer.allocateDirect(QUAD_VERTICES.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(QUAD_VERTICES)
            .also { it.position(0) }

        GLES30.glClearColor(0.0f, 0.0f, 0.0f, 1.0f)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        if (rendererFailed || width <= 0 || height <= 0) return
        GLES30.glViewport(0, 0, width, height)
        screenWidth = width
        screenHeight = height
        surfaceTexture?.setDefaultBufferSize(sourceWidth, sourceHeight)

        // Preserve the decoder's resolution for EASU and for linear fallback.
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, fboTextureId)
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA8, sourceWidth, sourceHeight, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboId)
        GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, fboTextureId, 0)
        if (GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER) != GLES30.GL_FRAMEBUFFER_COMPLETE) {
            failRenderer()
            return
        }

        fsrActive = fsrProgram != 0 && width > sourceWidth && height > sourceHeight
        if (fsrActive) {
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, fsrTextureId)
            GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA8, width, height, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fsrFboId)
            GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, fsrTextureId, 0)
            if (GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER) != GLES30.GL_FRAMEBUFFER_COMPLETE) {
                Log.w(TAG, "FSR framebuffer unavailable; using linear scaling")
                fsrActive = false
            }
        }
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
    }

    override fun onDrawFrame(gl: GL10?) {
        if (rendererFailed || !frameAvailable) return
        frameAvailable = false
        try {
            surfaceTexture?.updateTexImage()
            surfaceTexture?.getTransformMatrix(stMatrix)
        } catch (e: RuntimeException) {
            Log.e(TAG, "Video surface became unavailable", e)
            failRenderer()
            return
        }

        // --- PASS 1: OES to FBO ---
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboId)
        GLES30.glViewport(0, 0, sourceWidth, sourceHeight)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        GLES30.glUseProgram(copyProgram)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTextureId)
        GLES30.glUniform1i(uCopyTextureLoc, 0)
        GLES30.glUniformMatrix4fv(uCopySTMatrixLoc, 1, false, stMatrix, 0)

        vertexBuffer?.position(0)
        GLES30.glVertexAttribPointer(aCopyPosLoc, 2, GLES30.GL_FLOAT, false, VERTEX_STRIDE, vertexBuffer)
        GLES30.glEnableVertexAttribArray(aCopyPosLoc)
        vertexBuffer?.position(2)
        GLES30.glVertexAttribPointer(aCopyTexLoc, 2, GLES30.GL_FLOAT, false, VERTEX_STRIDE, vertexBuffer)
        GLES30.glEnableVertexAttribArray(aCopyTexLoc)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
        if (GLES30.glGetError() != GLES30.GL_NO_ERROR) {
            Log.e(TAG, "Video copy pass failed")
            failRenderer()
            return
        }

        // --- PASS 2: EASU from source resolution to display resolution ---
        if (fsrActive) {
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fsrFboId)
            GLES30.glViewport(0, 0, screenWidth, screenHeight)
            GLES30.glUseProgram(fsrProgram)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, fboTextureId)
            GLES30.glUniform1i(uFsrTextureLoc, 0)
            GLES30.glUniform2f(uFsrOutputSizeLoc, screenWidth.toFloat(), screenHeight.toFloat())
            vertexBuffer?.position(0)
            GLES30.glVertexAttribPointer(aFsrPosLoc, 2, GLES30.GL_FLOAT, false, VERTEX_STRIDE, vertexBuffer)
            GLES30.glEnableVertexAttribArray(aFsrPosLoc)
            vertexBuffer?.position(2)
            GLES30.glVertexAttribPointer(aFsrTexLoc, 2, GLES30.GL_FLOAT, false, VERTEX_STRIDE, vertexBuffer)
            GLES30.glEnableVertexAttribArray(aFsrTexLoc)
            GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
            GLES30.glDisableVertexAttribArray(aFsrPosLoc)
            GLES30.glDisableVertexAttribArray(aFsrTexLoc)
            val fsrError = GLES30.glGetError()
            if (fsrError != GLES30.GL_NO_ERROR) {
                Log.w(TAG, "FSR draw failed ($fsrError); using linear scaling")
                fsrActive = false
            }
        }

        // --- PASS 3: optional deband and sharpening, or linear presentation ---
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glViewport(0, 0, screenWidth, screenHeight)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        GLES30.glUseProgram(effectProgram)

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, if (fsrActive) fsrTextureId else fboTextureId)
        GLES30.glUniform1i(uEffTextureLoc, 0)
        GLES30.glUniform1f(uEffTimeLoc, frameCount)
        GLES30.glUniform2f(uEffScreenSizeLoc,
            if (fsrActive) screenWidth.toFloat() else sourceWidth.toFloat(),
            if (fsrActive) screenHeight.toFloat() else sourceHeight.toFloat())
        GLES30.glUniform1i(uEffDebandLoc, if (debandingEnabled) 1 else 0)
        GLES30.glUniform1f(uEffSharpnessLoc,
            if (fsrActive && rcasEnabled) 0.5f else if (!fsrActive && debandingEnabled) sharpness else 0f)

        vertexBuffer?.position(0)
        GLES30.glVertexAttribPointer(aEffPosLoc, 2, GLES30.GL_FLOAT, false, VERTEX_STRIDE, vertexBuffer)
        GLES30.glEnableVertexAttribArray(aEffPosLoc)
        vertexBuffer?.position(2)
        GLES30.glVertexAttribPointer(aEffTexLoc, 2, GLES30.GL_FLOAT, false, VERTEX_STRIDE, vertexBuffer)
        GLES30.glEnableVertexAttribArray(aEffTexLoc)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
        if (GLES30.glGetError() != GLES30.GL_NO_ERROR) {
            Log.e(TAG, "Video effect pass failed")
            failRenderer()
            return
        }

        frameCount += 1.0f
        if (frameCount > 1000000f) frameCount = 0f

        GLES30.glDisableVertexAttribArray(aEffPosLoc)
        GLES30.glDisableVertexAttribArray(aEffTexLoc)
    }

    override fun onFrameAvailable(surfaceTexture: SurfaceTexture?) {
        frameAvailable = true
        requestRender()
    }

    fun release() {
        surfaceTexture?.setOnFrameAvailableListener(null)
        surface?.release()
        surfaceTexture?.release()
        surface = null
        surfaceTexture = null
        // EGL owns the GL objects once GLSurfaceView tears down its context.
    }

    private fun failRenderer() {
        if (rendererFailed) return
        rendererFailed = true
        onRendererFailure()
    }

    private fun createProgram(vertexSource: String, fragmentSource: String): Int {
        val vertexShader = loadShader(GLES30.GL_VERTEX_SHADER, vertexSource)
        if (vertexShader == 0) return 0

        val fragmentShader = loadShader(GLES30.GL_FRAGMENT_SHADER, fragmentSource)
        if (fragmentShader == 0) {
            GLES30.glDeleteShader(vertexShader)
            return 0
        }

        val program = GLES30.glCreateProgram()
        if (program == 0) {
            GLES30.glDeleteShader(vertexShader)
            GLES30.glDeleteShader(fragmentShader)
            return 0
        }

        GLES30.glAttachShader(program, vertexShader)
        GLES30.glAttachShader(program, fragmentShader)
        GLES30.glLinkProgram(program)
        GLES30.glDeleteShader(vertexShader)
        GLES30.glDeleteShader(fragmentShader)

        val linkStatus = IntArray(1)
        GLES30.glGetProgramiv(program, GLES30.GL_LINK_STATUS, linkStatus, 0)
        if (linkStatus[0] != GLES30.GL_TRUE) {
            val log = GLES30.glGetProgramInfoLog(program)
            GLES30.glDeleteProgram(program)
            Log.e(TAG, "Program link failed: $log")
            return 0
        }

        return program
    }

    private fun loadShader(type: Int, source: String): Int {
        val shader = GLES30.glCreateShader(type)
        if (shader == 0) return 0

        GLES30.glShaderSource(shader, source)
        GLES30.glCompileShader(shader)

        val compileStatus = IntArray(1)
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, compileStatus, 0)
        if (compileStatus[0] != GLES30.GL_TRUE) {
            val log = GLES30.glGetShaderInfoLog(shader)
            GLES30.glDeleteShader(shader)
            Log.e(TAG, "Shader compile failed: $log")
            return 0
        }

        return shader
    }
}
