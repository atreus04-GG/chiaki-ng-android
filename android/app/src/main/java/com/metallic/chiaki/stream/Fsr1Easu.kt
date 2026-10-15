// SPDX-License-Identifier: MIT
//
// EASU filter adapted from AMD FidelityFX Super Resolution 1 (v1.20210629).
// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package com.metallic.chiaki.stream

/** FSR 1 EASU's twelve-tap edge-adaptive spatial upsampler for GLES 3.0. */
internal object Fsr1Easu {
    const val FRAGMENT_SHADER = """
        #version 300 es
        precision highp float;
        precision highp int;
        uniform sampler2D u_Texture;
        uniform vec2 u_OutputSize;
        out vec4 outColor;

        vec3 sampleTap(ivec2 base, ivec2 offset) {
            ivec2 size = textureSize(u_Texture, 0);
            return texelFetch(u_Texture, clamp(base + offset, ivec2(0), size - 1), 0).rgb;
        }

        float luma2(vec3 color) {
            return color.g + 0.5 * (color.r + color.b);
        }

        void setDirection(inout vec2 dir, inout float len,
                          float weight, float a, float b, float c, float d, float e) {
            float dx = d - b;
            float dy = e - a;
            float lx = clamp(abs(dx) / max(max(abs(d - c), abs(c - b)), 1e-6), 0.0, 1.0);
            float ly = clamp(abs(dy) / max(max(abs(e - c), abs(c - a)), 1e-6), 0.0, 1.0);
            dir += vec2(dx, dy) * weight;
            len += (lx * lx + ly * ly) * weight;
        }

        void addTap(inout vec3 color, inout float weight, vec2 offset,
                    vec2 dir, vec2 lengthScale, float lobe, float clipPoint,
                    vec3 tapColor) {
            vec2 rotated = vec2(offset.x * dir.x + offset.y * dir.y,
                                -offset.x * dir.y + offset.y * dir.x);
            rotated *= lengthScale;
            float distance2 = min(dot(rotated, rotated), clipPoint);
            float base = 0.4 * distance2 - 1.0;
            float window = lobe * distance2 - 1.0;
            float w = (1.5625 * base * base - 0.5625) * window * window;
            color += tapColor * w;
            weight += w;
        }

        void main() {
            vec2 sourceSize = vec2(textureSize(u_Texture, 0));
            vec2 position = gl_FragCoord.xy * sourceSize / u_OutputSize - 0.5;
            ivec2 base = ivec2(floor(position));
            vec2 pp = fract(position);

            vec3 b = sampleTap(base, ivec2( 0, -1));
            vec3 c = sampleTap(base, ivec2( 1, -1));
            vec3 e = sampleTap(base, ivec2(-1,  0));
            vec3 f = sampleTap(base, ivec2( 0,  0));
            vec3 g = sampleTap(base, ivec2( 1,  0));
            vec3 h = sampleTap(base, ivec2( 2,  0));
            vec3 i = sampleTap(base, ivec2(-1,  1));
            vec3 j = sampleTap(base, ivec2( 0,  1));
            vec3 k = sampleTap(base, ivec2( 1,  1));
            vec3 l = sampleTap(base, ivec2( 2,  1));
            vec3 n = sampleTap(base, ivec2( 0,  2));
            vec3 o = sampleTap(base, ivec2( 1,  2));

            float bl = luma2(b), cl = luma2(c), el = luma2(e);
            float fl = luma2(f), gl = luma2(g), hl = luma2(h);
            float il = luma2(i), jl = luma2(j), kl = luma2(k);
            float ll = luma2(l), nl = luma2(n), ol = luma2(o);
            vec2 dir = vec2(0.0);
            float len = 0.0;
            setDirection(dir, len, (1.0 - pp.x) * (1.0 - pp.y), bl, el, fl, gl, jl);
            setDirection(dir, len, pp.x * (1.0 - pp.y), cl, fl, gl, hl, kl);
            setDirection(dir, len, (1.0 - pp.x) * pp.y, fl, il, jl, kl, nl);
            setDirection(dir, len, pp.x * pp.y, gl, jl, kl, ll, ol);

            float magnitude = dot(dir, dir);
            dir = magnitude < 1.0 / 32768.0 ? vec2(1.0, 0.0) : dir * inversesqrt(magnitude);
            len = 0.25 * len * len;
            float stretch = dot(dir, dir) / max(max(abs(dir.x), abs(dir.y)), 1e-6);
            vec2 lengthScale = vec2(mix(1.0, stretch, len), 1.0 - 0.5 * len);
            float lobe = mix(0.5, 0.21, len);
            float clipPoint = 1.0 / lobe;

            vec3 minimum = min(min(f, g), min(j, k));
            vec3 maximum = max(max(f, g), max(j, k));
            vec3 color = vec3(0.0);
            float weight = 0.0;
            addTap(color, weight, vec2( 0.0, -1.0) - pp, dir, lengthScale, lobe, clipPoint, b);
            addTap(color, weight, vec2( 1.0, -1.0) - pp, dir, lengthScale, lobe, clipPoint, c);
            addTap(color, weight, vec2(-1.0,  1.0) - pp, dir, lengthScale, lobe, clipPoint, i);
            addTap(color, weight, vec2( 0.0,  1.0) - pp, dir, lengthScale, lobe, clipPoint, j);
            addTap(color, weight, vec2( 0.0,  0.0) - pp, dir, lengthScale, lobe, clipPoint, f);
            addTap(color, weight, vec2(-1.0,  0.0) - pp, dir, lengthScale, lobe, clipPoint, e);
            addTap(color, weight, vec2( 1.0,  1.0) - pp, dir, lengthScale, lobe, clipPoint, k);
            addTap(color, weight, vec2( 2.0,  1.0) - pp, dir, lengthScale, lobe, clipPoint, l);
            addTap(color, weight, vec2( 2.0,  0.0) - pp, dir, lengthScale, lobe, clipPoint, h);
            addTap(color, weight, vec2( 1.0,  0.0) - pp, dir, lengthScale, lobe, clipPoint, g);
            addTap(color, weight, vec2( 1.0,  2.0) - pp, dir, lengthScale, lobe, clipPoint, o);
            addTap(color, weight, vec2( 0.0,  2.0) - pp, dir, lengthScale, lobe, clipPoint, n);
            outColor = vec4(clamp(color / max(weight, 1e-6), minimum, maximum), 1.0);
        }
    """
}
