/*
**   VapourSynth port by HolyWu
**
**                TTempSmooth v0.9.4 for AviSynth 2.5.x
**
**   TTempSmooth is a basic, motion adaptive, temporal smoothing filter.
**   It currently supports YV12 and YUY2 colorspaces.
**
**   Copyright (C) 2004-2005 Kevin Stone
**
**   This program is free software: you can redistribute it and/or modify
**   it under the terms of the GNU General Public License as published by
**   the Free Software Foundation, either version 3 of the License, or
**   (at your option) any later version.
**
**   This program is distributed in the hope that it will be useful,
**   but WITHOUT ANY WARRANTY; without even the implied warranty of
**   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**   GNU General Public License for more details.
**
**   You should have received a copy of the GNU General Public License
**   along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <cmath>
#include <cstdlib>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <VapourSynth4.h>
#include <VSHelper4.h>

using namespace std::literals;

struct TTempSmoothData {
    VSNode* node;
    const VSVideoInfo* vi;
    int maxr;
    int thresh[3];
    int mdiff[3];
    double scthresh;
    bool fp;
    VSNode* pfclip;
    bool process[3];
    int shift;
    int diameter;
    float threshF[3];
    float* weight[3];
    float cw;
    VSNode* scNode;
    void (*filter[3])(const VSFrame* src[15], const VSFrame* pf[15], VSFrame* dst, const int fromFrame, const int toFrame, const int plane, const TTempSmoothData* const VS_RESTRICT d, const VSAPI* vsapi);
};

template<typename T>
static T getArg(const VSAPI* vsapi, const VSMap* map, const char* key, const T defaultValue) noexcept {
    T arg{};
    auto err{ 0 };

    if constexpr (std::is_same_v<T, bool>)
        arg = !!vsapi->mapGetInt(map, key, 0, &err);
    else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, long>)
        arg = vsapi->mapGetIntSaturated(map, key, 0, &err);
    else if constexpr (std::is_same_v<T, int64_t>)
        arg = vsapi->mapGetInt(map, key, 0, &err);
    else if constexpr (std::is_same_v<T, float>)
        arg = vsapi->mapGetFloatSaturated(map, key, 0, &err);
    else if constexpr (std::is_same_v<T, double>)
        arg = vsapi->mapGetFloat(map, key, 0, &err);

    if (err)
        arg = defaultValue;

    return arg;
}

template<typename T, bool useDiff>
static void filterI(const VSFrame* src[15], const VSFrame* pf[15], VSFrame* dst, const int fromFrame, const int toFrame, const int plane,
                    const TTempSmoothData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept {
    const auto width{ vsapi->getFrameWidth(dst, plane) };
    const auto height{ vsapi->getFrameHeight(dst, plane) };
    const auto stride{ vsapi->getStride(dst, plane) / sizeof(T) };
    const T* srcp[15]{};
    const T* pfp[15]{};
    for (auto i{ 0 }; i < d->diameter; i++) {
        srcp[i] = reinterpret_cast<const T*>(vsapi->getReadPtr(src[i], plane));
        pfp[i] = reinterpret_cast<const T*>(vsapi->getReadPtr(pf[i], plane));
    }
    T* VS_RESTRICT dstp{ reinterpret_cast<T*>(vsapi->getWritePtr(dst, plane)) };

    const auto thresh{ d->thresh[plane] };
    const auto* const weightSaved{ d->weight[plane] };

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++) {
            auto c{ pfp[d->maxr][x] };
            auto weights{ d->cw };
            auto sum{ srcp[d->maxr][x] * d->cw };

            auto frameIndex{ d->maxr - 1 };

            if (frameIndex > fromFrame) {
                auto t1{ pfp[frameIndex][x] };
                auto diff{ std::abs(c - t1) };

                if (diff < thresh) {
                    auto weight{ weightSaved[useDiff ? diff >> d->shift : frameIndex] };
                    weights += weight;
                    sum += srcp[frameIndex][x] * weight;

                    frameIndex--;
                    auto v{ 256 };

                    while (frameIndex > fromFrame) {
                        auto t2{ t1 };
                        t1 = pfp[frameIndex][x];
                        diff = std::abs(c - t1);

                        if (diff < thresh && std::abs(t1 - t2) < thresh) {
                            weight = weightSaved[useDiff ? (diff >> d->shift) + v : frameIndex];
                            weights += weight;
                            sum += srcp[frameIndex][x] * weight;

                            frameIndex--;
                            v += 256;
                        } else {
                            break;
                        }
                    }
                }
            }

            frameIndex = d->maxr + 1;

            if (frameIndex < toFrame) {
                auto t1{ pfp[frameIndex][x] };
                auto diff{ std::abs(c - t1) };

                if (diff < thresh) {
                    auto weight{ weightSaved[useDiff ? diff >> d->shift : frameIndex] };
                    weights += weight;
                    sum += srcp[frameIndex][x] * weight;

                    frameIndex++;
                    auto v{ 256 };

                    while (frameIndex < toFrame) {
                        auto t2{ t1 };
                        t1 = pfp[frameIndex][x];
                        diff = std::abs(c - t1);

                        if (diff < thresh && std::abs(t1 - t2) < thresh) {
                            weight = weightSaved[useDiff ? (diff >> d->shift) + v : frameIndex];
                            weights += weight;
                            sum += srcp[frameIndex][x] * weight;

                            frameIndex++;
                            v += 256;
                        } else {
                            break;
                        }
                    }
                }
            }

            if (d->fp)
                dstp[x] = static_cast<T>(srcp[d->maxr][x] * (1.0f - weights) + sum + 0.5f);
            else
                dstp[x] = static_cast<T>(sum / weights + 0.5f);
        }

        for (auto i{ 0 }; i < d->diameter; i++) {
            srcp[i] += stride;
            pfp[i] += stride;
        }
        dstp += stride;
    }
}

template<bool useDiff>
static void filterF(const VSFrame* src[15], const VSFrame* pf[15], VSFrame* dst, const int fromFrame, const int toFrame, const int plane,
                    const TTempSmoothData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept {
    const auto width{ vsapi->getFrameWidth(dst, plane) };
    const auto height{ vsapi->getFrameHeight(dst, plane) };
    const auto stride{ vsapi->getStride(dst, plane) / sizeof(float) };
    const float* srcp[15]{};
    const float* pfp[15]{};
    for (auto i{ 0 }; i < d->diameter; i++) {
        srcp[i] = reinterpret_cast<const float*>(vsapi->getReadPtr(src[i], plane));
        pfp[i] = reinterpret_cast<const float*>(vsapi->getReadPtr(pf[i], plane));
    }
    float* VS_RESTRICT dstp{ reinterpret_cast<float*>(vsapi->getWritePtr(dst, plane)) };

    const auto thresh{ d->threshF[plane] };
    const auto* const weightSaved{ d->weight[plane] };

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++) {
            auto c{ pfp[d->maxr][x] };
            auto weights{ d->cw };
            auto sum{ srcp[d->maxr][x] * d->cw };

            auto frameIndex{ d->maxr - 1 };

            if (frameIndex > fromFrame) {
                auto t1{ pfp[frameIndex][x] };
                auto diff{ std::min(std::abs(c - t1), 1.0f) };

                if (diff < thresh) {
                    auto weight{ weightSaved[useDiff ? static_cast<int>(diff * 255.0f) : frameIndex] };
                    weights += weight;
                    sum += srcp[frameIndex][x] * weight;

                    frameIndex--;
                    auto v{ 256 };

                    while (frameIndex > fromFrame) {
                        auto t2{ t1 };
                        t1 = pfp[frameIndex][x];
                        diff = std::min(std::abs(c - t1), 1.0f);

                        if (diff < thresh && std::min(std::abs(t1 - t2), 1.0f) < thresh) {
                            weight = weightSaved[useDiff ? static_cast<int>(diff * 255.0f) + v : frameIndex];
                            weights += weight;
                            sum += srcp[frameIndex][x] * weight;

                            frameIndex--;
                            v += 256;
                        } else {
                            break;
                        }
                    }
                }
            }

            frameIndex = d->maxr + 1;

            if (frameIndex < toFrame) {
                auto t1{ pfp[frameIndex][x] };
                auto diff{ std::min(std::abs(c - t1), 1.0f) };

                if (diff < thresh) {
                    auto weight{ weightSaved[useDiff ? static_cast<int>(diff * 255.0f) : frameIndex] };
                    weights += weight;
                    sum += srcp[frameIndex][x] * weight;

                    frameIndex++;
                    auto v{ 256 };

                    while (frameIndex < toFrame) {
                        auto t2{ t1 };
                        t1 = pfp[frameIndex][x];
                        diff = std::min(std::abs(c - t1), 1.0f);

                        if (diff < thresh && std::min(std::abs(t1 - t2), 1.0f) < thresh) {
                            weight = weightSaved[useDiff ? static_cast<int>(diff * 255.0f) + v : frameIndex];
                            weights += weight;
                            sum += srcp[frameIndex][x] * weight;

                            frameIndex++;
                            v += 256;
                        } else {
                            break;
                        }
                    }
                }
            }

            if (d->fp)
                dstp[x] = srcp[d->maxr][x] * (1.0f - weights) + sum;
            else
                dstp[x] = sum / weights;
        }

        for (auto i{ 0 }; i < d->diameter; i++) {
            srcp[i] += stride;
            pfp[i] += stride;
        }
        dstp += stride;
    }
}

static void selectFunctions(TTempSmoothData* d) noexcept {
    for (auto plane{ 0 }; plane < d->vi->format.numPlanes; plane++) {
        if (d->process[plane]) {
            if (d->thresh[plane] > d->mdiff[plane] + 1) {
                if (d->vi->format.bytesPerSample == 1)
                    d->filter[plane] = filterI<uint8_t, true>;
                else if (d->vi->format.bytesPerSample == 2)
                    d->filter[plane] = filterI<uint16_t, true>;
                else
                    d->filter[plane] = filterF<true>;
            } else {
                if (d->vi->format.bytesPerSample == 1)
                    d->filter[plane] = filterI<uint8_t, false>;
                else if (d->vi->format.bytesPerSample == 2)
                    d->filter[plane] = filterI<uint16_t, false>;
                else
                    d->filter[plane] = filterF<false>;
            }
        }
    }
}

static const VSFrame* VS_CC ttempsmoothGetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const TTempSmoothData*>(instanceData) };

    if (activationReason == arInitial) {
        for (auto i{ std::max(n - d->maxr, 0) }; i <= std::min(n + d->maxr, d->vi->numFrames - 1); i++) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);

            if (d->pfclip)
                vsapi->requestFrameFilter(i, d->pfclip, frameCtx);

            if (d->scthresh)
                vsapi->requestFrameFilter(i, d->scNode, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        const VSFrame* src[15]{};
        const VSFrame* pf[15]{};
        const VSFrame* sc[15]{};
        for (auto i{ n - d->maxr }; i <= n + d->maxr; i++) {
            auto frameNumber{ std::clamp(i, 0, d->vi->numFrames - 1) };

            src[i - n + d->maxr] = vsapi->getFrameFilter(frameNumber, d->node, frameCtx);

            if (d->pfclip)
                pf[i - n + d->maxr] = vsapi->getFrameFilter(frameNumber, d->pfclip, frameCtx);

            if (d->scthresh)
                sc[i - n + d->maxr] = vsapi->getFrameFilter(frameNumber, d->scNode, frameCtx);
        }
        const VSFrame* fr[]{ d->process[0] ? nullptr : src[d->maxr], d->process[1] ? nullptr : src[d->maxr], d->process[2] ? nullptr : src[d->maxr] };
        const int pl[]{ 0, 1, 2 };
        auto dst{ vsapi->newVideoFrame2(&d->vi->format, d->vi->width, d->vi->height, fr, pl, src[d->maxr], core) };

        auto fromFrame{ -1 };
        auto toFrame{ d->diameter };
        if (d->scthresh) {
            for (auto i{ d->maxr }; i > 0; i--) {
                if (vsapi->mapGetInt(vsapi->getFramePropertiesRO(sc[i]), "_SceneChangePrev", 0, nullptr)) {
                    fromFrame = i;
                    break;
                }
            }

            for (auto i{ d->maxr }; i < d->diameter - 1; i++) {
                if (vsapi->mapGetInt(vsapi->getFramePropertiesRO(sc[i]), "_SceneChangeNext", 0, nullptr)) {
                    toFrame = i;
                    break;
                }
            }
        }

        for (auto plane{ 0 }; plane < d->vi->format.numPlanes; plane++) {
            if (d->process[plane])
                d->filter[plane](src, d->pfclip ? pf : src, dst, fromFrame, toFrame, plane, d, vsapi);
        }

        for (auto i{ 0 }; i < d->diameter; i++) {
            vsapi->freeFrame(src[i]);
            vsapi->freeFrame(pf[i]);
            vsapi->freeFrame(sc[i]);
        }

        return dst;
    }

    return nullptr;
}

static void VS_CC ttempsmoothFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<TTempSmoothData*>(instanceData) };

    vsapi->freeNode(d->node);
    vsapi->freeNode(d->pfclip);
    vsapi->freeNode(d->scNode);

    for (auto i{ 0 }; i < 3; i++)
        delete[] d->weight[i];

    delete d;
}

static void VS_CC ttempsmoothCreate(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto d{ std::make_unique<TTempSmoothData>() };

    try {
        auto err{ 0 };

        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->pfclip = vsapi->mapGetNode(in, "pfclip", 0, &err);
        d->vi = vsapi->getVideoInfo(d->node);

        if (!vsh::isConstantVideoFormat(d->vi) ||
            (d->vi->format.sampleType == stInteger && d->vi->format.bitsPerSample > 16) ||
            (d->vi->format.sampleType == stFloat && d->vi->format.bitsPerSample != 32))
            throw "only constant format 8-16 bit integer and 32 bit float input supported";

        d->maxr = getArg(vsapi, in, "maxr", 3);
        auto strength = getArg(vsapi, in, "strength", 2);
        d->scthresh = getArg(vsapi, in, "scthresh", 12.0);
        d->fp = getArg(vsapi, in, "fp", true);

        auto numThresh{ vsapi->mapNumElements(in, "thresh") };
        if (numThresh > d->vi->format.numPlanes)
            throw "more thresh given than there are planes";

        for (auto i{ 0 }; i < numThresh; i++)
            d->thresh[i] = vsapi->mapGetIntSaturated(in, "thresh", i, nullptr);
        if (numThresh <= 0) {
            d->thresh[0] = 4;
            d->thresh[1] = d->thresh[2] = 5;
        } else if (numThresh == 1) {
            d->thresh[1] = d->thresh[2] = d->thresh[0];
        } else if (numThresh == 2) {
            d->thresh[2] = d->thresh[1];
        }

        auto numMdiff{ vsapi->mapNumElements(in, "mdiff") };
        if (numMdiff > d->vi->format.numPlanes)
            throw "more mdiff given than there are planes";

        for (auto i{ 0 }; i < numMdiff; i++)
            d->mdiff[i] = vsapi->mapGetIntSaturated(in, "mdiff", i, nullptr);
        if (numMdiff <= 0) {
            d->mdiff[0] = 2;
            d->mdiff[1] = d->mdiff[2] = 3;
        } else if (numMdiff == 1) {
            d->mdiff[1] = d->mdiff[2] = d->mdiff[0];
        } else if (numMdiff == 2) {
            d->mdiff[2] = d->mdiff[1];
        }

        auto m{ vsapi->mapNumElements(in, "planes") };

        for (auto i{ 0 }; i < 3; i++)
            d->process[i] = (m <= 0);

        for (auto i{ 0 }; i < m; i++) {
            auto n{ vsapi->mapGetIntSaturated(in, "planes", i, nullptr) };

            if (n < 0 || n >= d->vi->format.numPlanes)
                throw "plane index out of range";

            if (d->process[n])
                throw "plane specified twice";

            d->process[n] = true;
        }

        if (d->maxr < 1 || d->maxr > 7)
            throw "maxr must be between 1 and 7 (inclusive)";

        for (auto i{ 0 }; i < 3; i++) {
            if (d->thresh[i] < 1 || d->thresh[i] > 256)
                throw "thresh must be between 1 and 256 (inclusive)";

            if (d->mdiff[i] < 0 || d->mdiff[i] > 255)
                throw "mdiff must be between 0 and 255 (inclusive)";
        }

        if (strength < 1 || strength > 8)
            throw "strength must be 1, 2, 3, 4, 5, 6, 7, or 8";

        if (d->scthresh < 0.0 || d->scthresh > 100.0)
            throw "scthresh must be between 0.0 and 100.0 (inclusive)";

        if (d->pfclip) {
            if (!vsh::isSameVideoInfo(vsapi->getVideoInfo(d->pfclip), d->vi))
                throw "pfclip must have the same format and dimensions as main clip";

            if (vsapi->getVideoInfo(d->pfclip)->numFrames != d->vi->numFrames)
                throw "pfclip's number of frames does not match";
        }

        selectFunctions(d.get());

        if (d->vi->format.sampleType == stInteger)
            d->shift = d->vi->format.bitsPerSample - 8;

        d->diameter = d->maxr * 2 + 1;

        for (auto plane{ 0 }; plane < d->vi->format.numPlanes; plane++) {
            if (d->process[plane]) {
                if (d->thresh[plane] > d->mdiff[plane] + 1) {
                    d->weight[plane] = new float[256 * d->maxr];
                    float dt[15]{};
                    float rt[256]{};
                    float sum{};

                    for (auto i{ 0 }; i < strength && i <= d->maxr; i++)
                        dt[i] = 1.0f;
                    for (auto i{ strength }; i <= d->maxr; i++)
                        dt[i] = 1.0f / (i - strength + 2);

                    auto step{ 256.0f / (d->thresh[plane] - std::min(d->mdiff[plane], d->thresh[plane] - 1)) };
                    auto base{ 256.0f };
                    for (auto i{ 0 }; i < d->thresh[plane]; i++) {
                        if (d->mdiff[plane] > i) {
                            rt[i] = 256.0f;
                        } else {
                            if (base > 0.0f)
                                rt[i] = base;
                            else
                                break;
                            base -= step;
                        }
                    }

                    sum += dt[0];
                    for (auto i{ 1 }; i <= d->maxr; i++) {
                        sum += dt[i] * 2.0f;
                        for (auto v{ 0 }; v < 256; v++)
                            d->weight[plane][256 * (i - 1) + v] = dt[i] * rt[v] / 256.0f;
                    }

                    for (auto i{ 0 }; i < 256 * d->maxr; i++)
                        d->weight[plane][i] /= sum;

                    d->cw = dt[0] / sum;
                } else {
                    d->weight[plane] = new float[d->diameter];
                    float dt[15]{};
                    float sum{};

                    for (auto i{ 0 }; i < strength && i <= d->maxr; i++)
                        dt[d->maxr - i] = dt[d->maxr + i] = 1.0f;
                    for (auto i{ strength }; i <= d->maxr; i++)
                        dt[d->maxr - i] = dt[d->maxr + i] = 1.0f / (i - strength + 2);

                    for (auto i{ 0 }; i < d->diameter; i++) {
                        sum += dt[i];
                        d->weight[plane][i] = dt[i];
                    }

                    for (auto i{ 0 }; i < d->diameter; i++)
                        d->weight[plane][i] /= sum;

                    d->cw = d->weight[plane][d->maxr];
                }

                if (d->vi->format.sampleType == stInteger)
                    d->thresh[plane] <<= d->shift;
                else
                    d->threshF[plane] = d->thresh[plane] / 256.0f;
            }
        }

        if (d->scthresh) {
            auto miscPlugin{ vsapi->getPluginByID("com.vapoursynth.misc", core) };
            if (!miscPlugin)
                throw "MiscFilters (https://github.com/vapoursynth/vs-miscfilters-obsolete) not installed";

            auto args{ vsapi->createMap() };

            if (d->vi->format.colorFamily == cfRGB) {
                vsapi->mapSetNode(args, "clip", d->pfclip ? d->pfclip : d->node, maReplace);
                vsapi->mapSetInt(args, "format", pfGray8, maReplace);
                vsapi->mapSetData(args, "matrix_s", "709", -1, dtUtf8, maReplace);

                auto ret{ vsapi->invoke(vsapi->getPluginByID(VSH_RESIZE_PLUGIN_ID, core), "Point", args) };
                if (vsapi->mapGetError(ret)) {
                    vsapi->mapSetError(out, vsapi->mapGetError(ret));
                    vsapi->freeNode(d->node);
                    vsapi->freeNode(d->pfclip);
                    vsapi->freeMap(args);
                    vsapi->freeMap(ret);
                    return;
                }

                vsapi->clearMap(args);
                vsapi->mapConsumeNode(args, "clip", vsapi->mapGetNode(ret, "clip", 0, nullptr), maReplace);
                vsapi->freeMap(ret);
            } else {
                vsapi->mapSetNode(args, "clip", d->pfclip ? d->pfclip : d->node, maReplace);
            }
            vsapi->mapSetFloat(args, "threshold", d->scthresh / 100.0, maReplace);

            auto ret{ vsapi->invoke(miscPlugin, "SCDetect", args) };
            if (vsapi->mapGetError(ret)) {
                vsapi->mapSetError(out, vsapi->mapGetError(ret));
                vsapi->freeNode(d->node);
                vsapi->freeNode(d->pfclip);
                vsapi->freeMap(args);
                vsapi->freeMap(ret);
                return;
            }

            d->scNode = vsapi->mapGetNode(ret, "clip", 0, nullptr);
            vsapi->freeMap(args);
            vsapi->freeMap(ret);
        }
    } catch (const char* error) {
        vsapi->mapSetError(out, ("TTempSmooth: "s + error).c_str());
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->pfclip);
        return;
    }

    std::vector<VSFilterDependency> deps;
    deps.push_back({ d->node, rpGeneral });
    if (d->pfclip)
        deps.push_back({ d->pfclip, rpGeneral });
    if (d->scNode)
        deps.push_back({ d->scNode, rpGeneral });

    vsapi->createVideoFilter(out, "TTempSmooth", d->vi, ttempsmoothGetFrame, ttempsmoothFree, fmParallel, deps.data(), deps.size(), d.get(), core);
    d.release();
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.holywu.ttempsmooth", "ttmpsm", "A basic, motion adaptive, temporal smoothing filter", VS_MAKE_VERSION(4, 1), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("TTempSmooth",
                             "clip:vnode;"
                             "maxr:int:opt;"
                             "thresh:int[]:opt;"
                             "mdiff:int[]:opt;"
                             "strength:int:opt;"
                             "scthresh:float:opt;"
                             "fp:int:opt;"
                             "pfclip:vnode:opt;"
                             "planes:int[]:opt;",
                             "clip:vnode;",
                             ttempsmoothCreate, nullptr, plugin);
}
