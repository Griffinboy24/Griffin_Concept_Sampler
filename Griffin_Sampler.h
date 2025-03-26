#pragma once

#include <JuceHeader.h>
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <new>
#include <atomic>

namespace project
{

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

    using namespace juce;
    using namespace hise;
    using namespace scriptnode;

    static constexpr int FIXED_SHIFT = 16;
    static constexpr int64_t FIXED_ONE = (int64_t)1 << FIXED_SHIFT;
    static constexpr int64_t FIXED_MASK = FIXED_ONE - 1;

    struct SampleSettings
    {
        double pitchOffsetCents = 0.0;
        float  volumeMult = 1.0f;
        float  panning = 0.0f;
        float  playbackStartInSamples = 0.0f;
        float  loopStartInSamples = 0.0f;
        float  loopEndInSamples = 0.0f;
        bool   loopMode = false;
        float  xfadeLengthInSamples = 0.0f;
    };

    struct SamplePlayback
    {
        const float* sourceL = nullptr;
        const float* sourceR = nullptr;
        bool active = false;
        float amplitude = 1.0f;
        SampleSettings settings;
        int64_t phaseAcc = 0;
        int64_t phaseInc = 0;
        float playbackStart = 0.0f;
        float loopStart = 0.0f;
        float loopEnd = 0.0f;

        SamplePlayback() noexcept = default;

        SamplePlayback(const std::array<const float*, 2>& src, float amp,
            const SampleSettings& s, int bufferLength, double baseDelta)
        {
            sourceL = src[0];
            sourceR = src[1];
            settings = s;
            amplitude = amp * s.volumeMult;
            active = true;

            float pbStart = s.playbackStartInSamples;
            if (pbStart < 0.f)
                pbStart = 0.f;
            if (pbStart > float(bufferLength - 1))
                pbStart = float(bufferLength - 1);
            playbackStart = pbStart;

            float lStart = s.loopStartInSamples;
            float lEnd = s.loopEndInSamples;
            if (!s.loopMode)
            {
                lStart = pbStart;
                lEnd = s.loopEndInSamples;
            }
            else
            {
                if (lStart < 0.f)
                    lStart = 0.f;
                if (lStart > float(bufferLength - 1))
                    lStart = float(bufferLength - 1);
                if (lEnd < 0.f)
                    lEnd = 0.f;
                if (lEnd > float(bufferLength - 1))
                    lEnd = float(bufferLength - 1);
            }
            loopStart = lStart;
            loopEnd = lEnd;

            phaseAcc = (int64_t)std::llround(pbStart * FIXED_ONE);
            double centsFact = std::pow(2.0, (s.pitchOffsetCents / 1200.0));
            double effectiveSpeed = baseDelta * centsFact;
            phaseInc = (int64_t)std::llround(effectiveSpeed * FIXED_ONE);
        }

        // Update settings without resetting phaseAcc.
        FORCE_INLINE void updateSettings(const SampleSettings& s, int bufferLength)
        {
            settings = s;
            float pbStart = s.playbackStartInSamples;
            if (pbStart < 0.f)
                pbStart = 0.f;
            if (pbStart > float(bufferLength - 1))
                pbStart = float(bufferLength - 1);
            playbackStart = pbStart;
            float lStart = s.loopStartInSamples;
            float lEnd = s.loopEndInSamples;
            if (!s.loopMode)
            {
                lStart = pbStart;
                lEnd = s.loopEndInSamples;
            }
            else
            {
                if (lStart < 0.f)
                    lStart = 0.f;
                if (lStart > float(bufferLength - 1))
                    lStart = float(bufferLength - 1);
                if (lEnd < 0.f)
                    lEnd = 0.f;
                if (lEnd > float(bufferLength - 1))
                    lEnd = float(bufferLength - 1);
            }
            loopStart = lStart;
            loopEnd = lEnd;
        }

        // Highly optimized vectorSynthesize using fixed-point arithmetic and precomputed boundaries.
        int vectorSynthesize(float* outL, float* outR, int blockSize,
            const AudioBuffer<float>* preXfadeBuffer, float preXfadeLength)
        {
            if (!active)
                return 0;

            int processed = 0;
            const float invFixedOne = 1.f / FIXED_ONE;
            const float leftGain = 0.5f * (1.f - settings.panning);
            const float rightGain = 0.5f * (1.f + settings.panning);
            const float ampLeft = amplitude * leftGain;
            const float ampRight = amplitude * rightGain;
            const bool loopEnabled = settings.loopMode;
            const float X = settings.xfadeLengthInSamples;
            const float endSample = loopEnd;
            // For crossfade region, compute the float boundary.
            const float crossfadeStartF = loopEnabled ? (endSample - X) : 0.f;
            const float piOverTwo = float(M_PI * 0.5f);
            // Precompute fixed–point boundaries.
            const int64_t fixedEnd = int64_t(endSample * FIXED_ONE);
            int64_t fixedLoopStart = 0;
            int64_t fixedCrossfadeStart = 0;
            int64_t fixedX = 0;
            if (loopEnabled)
            {
                fixedLoopStart = int64_t(loopStart * FIXED_ONE);
                fixedX = (int64_t)std::llround(X * FIXED_ONE);
                fixedCrossfadeStart = fixedEnd - fixedX;
            }

            // Main processing loop.
            while (processed < blockSize)
            {
                // Process using phaseAcc in fixed point.
                if (!loopEnabled)
                {
                    if (phaseAcc >= fixedEnd)
                    {
                        active = false;
                        break;
                    }
                    // Compute number of samples until reaching fixedEnd.
                    int64_t samplesToBoundary = (fixedEnd - phaseAcc + phaseInc - 1) / phaseInc;
                    int n = (samplesToBoundary > (blockSize - processed)) ? (blockSize - processed) : int(samplesToBoundary);
                    for (int i = 0; i < n; i++)
                    {
                        int idx = int(phaseAcc >> FIXED_SHIFT);
                        float frac = float(phaseAcc & FIXED_MASK) * invFixedOne;
                        float sampL = sourceL[idx] + frac * (sourceL[idx + 1] - sourceL[idx]);
                        float sampR = sourceR[idx] + frac * (sourceR[idx + 1] - sourceR[idx]);
                        outL[processed + i] += sampL * ampLeft;
                        outR[processed + i] += sampR * ampRight;
                        phaseAcc += phaseInc;
                    }
                    processed += n;
                }
                else
                {
                    if (phaseAcc < fixedLoopStart)
                    {
                        // Process until entering loop region.
                        int64_t samplesToBoundary = (fixedLoopStart - phaseAcc + phaseInc - 1) / phaseInc;
                        int n = (samplesToBoundary > (blockSize - processed)) ? (blockSize - processed) : int(samplesToBoundary);
                        for (int i = 0; i < n; i++)
                        {
                            int idx = int(phaseAcc >> FIXED_SHIFT);
                            float frac = float(phaseAcc & FIXED_MASK) * invFixedOne;
                            float sampL = sourceL[idx] + frac * (sourceL[idx + 1] - sourceL[idx]);
                            float sampR = sourceR[idx] + frac * (sourceR[idx + 1] - sourceR[idx]);
                            outL[processed + i] += sampL * ampLeft;
                            outR[processed + i] += sampR * ampRight;
                            phaseAcc += phaseInc;
                        }
                        processed += n;
                    }
                    else if (phaseAcc >= fixedEnd)
                    {
                        // Wrap-around: compute excess and restart at loop start plus crossfade.
                        int64_t excess = phaseAcc - fixedEnd;
                        phaseAcc = fixedLoopStart + fixedX + excess;
                        continue;
                    }
                    else if (phaseAcc < fixedCrossfadeStart)
                    {
                        // Process normal playback until entering crossfade region.
                        int64_t samplesToBoundary = (fixedCrossfadeStart - phaseAcc + phaseInc - 1) / phaseInc;
                        int n = (samplesToBoundary > (blockSize - processed)) ? (blockSize - processed) : int(samplesToBoundary);
                        for (int i = 0; i < n; i++)
                        {
                            int idx = int(phaseAcc >> FIXED_SHIFT);
                            float frac = float(phaseAcc & FIXED_MASK) * invFixedOne;
                            float sampL = sourceL[idx] + frac * (sourceL[idx + 1] - sourceL[idx]);
                            float sampR = sourceR[idx] + frac * (sourceR[idx + 1] - sourceR[idx]);
                            outL[processed + i] += sampL * ampLeft;
                            outR[processed + i] += sampR * ampRight;
                            phaseAcc += phaseInc;
                        }
                        processed += n;
                    }
                    else
                    {
                        // Crossfade region.
                        int64_t samplesToBoundary = (fixedEnd - phaseAcc + phaseInc - 1) / phaseInc;
                        int n = (samplesToBoundary > (blockSize - processed)) ? (blockSize - processed) : int(samplesToBoundary);
                        if (preXfadeBuffer)
                        {
                            const float* xfadeL = preXfadeBuffer->getReadPointer(0);
                            const float* xfadeR = preXfadeBuffer->getReadPointer(1);
                            int xfadeBufferLen = preXfadeBuffer->getNumSamples();
                            for (int i = 0; i < n; i++)
                            {
                                int idxPhase = int(phaseAcc >> FIXED_SHIFT);
                                float frac = float(phaseAcc & FIXED_MASK) * invFixedOne;
                                // Convert current phase to float.
                                float currentP = float(phaseAcc) * invFixedOne;
                                float posInXfade = currentP - crossfadeStartF;
                                int idx = int(posInXfade);
                                float subFrac = posInXfade - idx;
                                if (idx < 0)
                                    idx = 0;
                                else if (idx >= xfadeBufferLen - 1)
                                    idx = xfadeBufferLen - 2;
                                float sampL = xfadeL[idx] + subFrac * (xfadeL[idx + 1] - xfadeL[idx]);
                                float sampR = xfadeR[idx] + subFrac * (xfadeR[idx + 1] - xfadeR[idx]);
                                outL[processed + i] += sampL * ampLeft;
                                outR[processed + i] += sampR * ampRight;
                                phaseAcc += phaseInc;
                            }
                        }
                        else
                        {
                            for (int i = 0; i < n; i++)
                            {
                                int idxPhase = int(phaseAcc >> FIXED_SHIFT);
                                float frac = float(phaseAcc & FIXED_MASK) * invFixedOne;
                                float currentP = float(phaseAcc) * invFixedOne;
                                float alpha = (currentP - crossfadeStartF) / X;
                                float crossAngle = alpha * piOverTwo;
                                float tailGain = std::cos(crossAngle);
                                float headGain = std::sin(crossAngle);
                                float sampTailL = sourceL[idxPhase] + frac * (sourceL[idxPhase + 1] - sourceL[idxPhase]);
                                float sampTailR = sourceR[idxPhase] + frac * (sourceR[idxPhase + 1] - sourceR[idxPhase]);
                                float headPos = float(fixedLoopStart) * invFixedOne + (currentP - crossfadeStartF);
                                int idxHead = int(headPos);
                                float fracHead = headPos - idxHead;
                                float sampHeadL = sourceL[idxHead] + fracHead * (sourceL[idxHead + 1] - sourceL[idxHead]);
                                float sampHeadR = sourceR[idxHead] + fracHead * (sourceR[idxHead + 1] - sourceR[idxHead]);
                                float mixL = tailGain * sampTailL + headGain * sampHeadL;
                                float mixR = tailGain * sampTailR + headGain * sampHeadR;
                                outL[processed + i] += mixL * ampLeft;
                                outR[processed + i] += mixR * ampRight;
                                phaseAcc += phaseInc;
                            }
                        }
                        processed += n;
                    }
                }
            }
            return processed;
        }
    };

    struct Voice
    {
        int midiNote = 60;
        bool isActive = false;
        float velocity = 1.0f;
        SamplePlayback playback;

        void reset(int note, float vel, const std::array<const float*, 2>& sample,
            int bufferLength, double baseDelta, const SampleSettings& settings)
        {
            midiNote = note;
            velocity = vel;
            isActive = true;
            new (&playback) SamplePlayback(sample, velocity, settings, bufferLength, baseDelta);
        }
    };

    template <int NV>
    struct Griffin_Sampler : public data::base
    {
        SNEX_NODE(Griffin_Sampler);
        struct MetadataClass { SN_NODE_ID("Griffin_Sampler"); };

        static constexpr bool isModNode() { return false; }
        static constexpr bool isPolyphonic() { return NV > 1; }
        static constexpr bool hasTail() { return true; }
        static constexpr bool isSuspendedOnSilence() { return false; }
        static constexpr int getFixChannelAmount() { return 2; }
        static constexpr int NumTables = 0;
        static constexpr int NumSliderPacks = 0;
        static constexpr int NumAudioFiles = 1;
        static constexpr int NumFilters = 0;
        static constexpr int NumDisplayBuffers = 0;

        PolyData<Voice, NV> voices;
        ExternalData sampleData;
        AudioBuffer<float> sampleBuffer;
        std::array<const float*, 2> sample{ nullptr, nullptr };

        std::array<float, 128> pitchRatios{};
        double sampleRate = 44100.0;
        double sampleRateRatio = 1.0;

        float sampleStartPercent = 0.0f;
        float loopStartPercent = 0.0f;
        float loopEndPercent = 1.0f;

        float playbackStartOffsetInSamples = 0.0f;
        float loopStartOffsetInSamples = 0.0f;
        float loopEndOffsetInSamples = 0.0f;

        double globalPitchOffsetFactor = 1.0;

        float xfadeFraction = 0.0f;
        float xfadeLengthInSamples = 0.0f;
        bool  loopMode = false;

        std::mt19937 randomGen;
        std::atomic<AudioBuffer<float>*> precomputedXfadeBuffer{ nullptr };

        void setExternalData(const ExternalData& ed, int)
        {
            sampleData = ed;
            AudioSampleBuffer tempBuf = ed.toAudioSampleBuffer();
            int numSamples = tempBuf.getNumSamples();
            int numChannels = tempBuf.getNumChannels();
            if (numSamples <= 0)
            {
                int fallbackLen = 8;
                int chs = (numChannels > 0 ? numChannels : 2);
                AudioSampleBuffer fallback(chs, fallbackLen);
                fallback.clear();
                sampleBuffer.makeCopyOf(fallback, true);
            }
            else
            {
                sampleBuffer.makeCopyOf(tempBuf, true);
            }
            sample[0] = sampleBuffer.getReadPointer(0);
            if (numChannels > 1)
                sample[1] = sampleBuffer.getReadPointer(1);
            else
                sample[1] = sample[0];

            updateDerivedParameters();
        }

        void updateDerivedParameters()
        {
            int currentSampleLength = sampleBuffer.getNumSamples();
            if (currentSampleLength < 1)
                currentSampleLength = 1;

            playbackStartOffsetInSamples = sampleStartPercent * float(currentSampleLength - 1);
            loopStartOffsetInSamples = loopStartPercent * float(currentSampleLength - 1);
            loopEndOffsetInSamples = loopEndPercent * float(currentSampleLength - 1);

            float regionLen = loopEndOffsetInSamples - loopStartOffsetInSamples;
            if (regionLen < 0.f)
                regionLen = 0.f;

            float maxXfade = regionLen * 0.5f;
            float desiredXfade = xfadeFraction * regionLen;
            if (desiredXfade > maxXfade)
                desiredXfade = maxXfade;
            xfadeLengthInSamples = desiredXfade;

            if (loopMode && xfadeLengthInSamples > 0.f && sampleBuffer.getNumSamples() > 0)
            {
                int xfadeSamples = std::max(1, (int)std::round(xfadeLengthInSamples));
                auto* newXfadeBuffer = new AudioBuffer<float>(2, xfadeSamples);
                for (int ch = 0; ch < 2; ++ch)
                {
                    float* dest = newXfadeBuffer->getWritePointer(ch);
                    const float* src = sampleBuffer.getReadPointer(std::min(ch, sampleBuffer.getNumChannels() - 1));
                    for (int i = 0; i < xfadeSamples; i++)
                    {
                        float pos = float(i);
                        float alpha = (xfadeSamples > 1 ? pos / float(xfadeSamples - 1) : 0.f);
                        float tailGain = std::cos(alpha * (float(M_PI) * 0.5f));
                        float headGain = std::sin(alpha * (float(M_PI) * 0.5f));
                        float tailPos = loopEndOffsetInSamples - xfadeLengthInSamples + pos;
                        float headPos = loopStartOffsetInSamples + pos;
                        int tailIdx = int(tailPos);
                        int headIdx = int(headPos);
                        float tailFrac = tailPos - tailIdx;
                        float headFrac = headPos - headIdx;
                        int numSamples = sampleBuffer.getNumSamples();
                        int tailIdx1 = std::min(tailIdx + 1, numSamples - 1);
                        int headIdx1 = std::min(headIdx + 1, numSamples - 1);
                        float tailSample = src[tailIdx] + tailFrac * (src[tailIdx1] - src[tailIdx]);
                        float headSample = src[headIdx] + headFrac * (src[headIdx1] - src[headIdx]);
                        dest[i] = tailGain * tailSample + headGain * headSample;
                    }
                }
                AudioBuffer<float>* oldBuffer = precomputedXfadeBuffer.exchange(newXfadeBuffer);
                if (oldBuffer)
                    delete oldBuffer;
            }
            else
            {
                AudioBuffer<float>* oldBuffer = precomputedXfadeBuffer.exchange(nullptr);
                if (oldBuffer)
                    delete oldBuffer;
            }
        }

        void reset()
        {
            for (auto& voice : voices)
                voice.isActive = false;
        }

        void prepare(PrepareSpecs specs)
        {
            sampleRate = specs.sampleRate;
            initPitchRatios();
            updateDerivedParameters();
            voices.prepare(specs);
            std::random_device rd;
            randomGen.seed(rd());
        }

        void handleHiseEvent(HiseEvent& e)
        {
            if (e.isNoteOn())
            {
                auto& voice = voices.get();
                double baseDelta = pitchRatios[e.getNoteNumber()] * sampleRateRatio * globalPitchOffsetFactor;
                SampleSettings settings;
                settings.pitchOffsetCents = 0.0;
                settings.volumeMult = 1.0f;
                settings.panning = 0.0f;
                settings.playbackStartInSamples = playbackStartOffsetInSamples;
                settings.loopStartInSamples = loopStartOffsetInSamples;
                settings.loopEndInSamples = loopEndOffsetInSamples;
                settings.loopMode = loopMode;
                settings.xfadeLengthInSamples = xfadeLengthInSamples;
                voice.reset(e.getNoteNumber(), e.getFloatVelocity(), sample,
                    sampleBuffer.getNumSamples(), baseDelta, settings);
            }
        }

        template <typename ProcessDataType>
        void process(ProcessDataType& data)
        {
            auto& fixData = data.template as<ProcessData<getFixChannelAmount()>>();
            auto audioBlock = fixData.toAudioBlock();
            auto* leftChannel = audioBlock.getChannelPointer(0);
            auto* rightChannel = audioBlock.getChannelPointer(1);
            int totalSamples = data.getNumSamples();

            if (sampleBuffer.getNumSamples() == 0)
            {
                audioBlock.clear();
                return;
            }

            std::fill(leftChannel, leftChannel + totalSamples, 0.f);
            std::fill(rightChannel, rightChannel + totalSamples, 0.f);

            AudioBuffer<float>* currentXfadeBuffer = precomputedXfadeBuffer.load();
            for (auto& voice : voices)
            {
                if (!voice.isActive)
                    continue;

                int n = voice.playback.vectorSynthesize(leftChannel, rightChannel, totalSamples,
                    currentXfadeBuffer, xfadeLengthInSamples);
                if (!voice.playback.active)
                    voice.isActive = false;
            }
        }

        template <typename FrameDataType>
        void processFrame(FrameDataType&) {}

        template <int P>
        void setParameter(double v)
        {
            if constexpr (P == 0)
            {
                globalPitchOffsetFactor = std::pow(2.0, v / 12.0);
                for (auto& voice : voices)
                {
                    if (voice.isActive)
                    {
                        double newBaseDelta = pitchRatios[voice.midiNote] * sampleRateRatio * globalPitchOffsetFactor;
                        double centsFact = std::pow(2.0, (voice.playback.settings.pitchOffsetCents / 1200.0));
                        double effectiveSpeed = newBaseDelta * centsFact;
                        voice.playback.phaseInc = (int64_t)std::llround(effectiveSpeed * FIXED_ONE);
                    }
                }
            }
            else if constexpr (P == 1)
            {
                sampleStartPercent = (float)v;
                updateDerivedParameters();
            }
            else if constexpr (P == 2)
            {
                loopStartPercent = (float)v;
                updateDerivedParameters();
            }
            else if constexpr (P == 3)
            {
                loopEndPercent = (float)v;
                updateDerivedParameters();
            }
            else if constexpr (P == 4)
            {
                xfadeFraction = (float)v;
                if (xfadeFraction < 0.f)
                    xfadeFraction = 0.f;
                if (xfadeFraction > 1.f)
                    xfadeFraction = 1.f;
                updateDerivedParameters();
            }
            else if constexpr (P == 5)
            {
                loopMode = (v >= 0.5);
                updateDerivedParameters();
            }
            if constexpr (P >= 1 && P <= 5)
            {
                for (auto& voice : voices)
                {
                    if (voice.isActive)
                    {
                        SampleSettings newSettings;
                        newSettings.pitchOffsetCents = voice.playback.settings.pitchOffsetCents;
                        newSettings.volumeMult = voice.playback.settings.volumeMult;
                        newSettings.panning = voice.playback.settings.panning;
                        newSettings.playbackStartInSamples = playbackStartOffsetInSamples;
                        newSettings.loopStartInSamples = loopStartOffsetInSamples;
                        newSettings.loopEndInSamples = loopEndOffsetInSamples;
                        newSettings.loopMode = loopMode;
                        newSettings.xfadeLengthInSamples = xfadeLengthInSamples;
                        voice.playback.updateSettings(newSettings, sampleBuffer.getNumSamples());
                    }
                }
            }
        }

        void initPitchRatios()
        {
            for (int i = 0; i < 128; ++i)
                pitchRatios[i] = std::pow(2.0f, float(i - 60) / 12.0f);
        }

        void createParameters(ParameterDataList& data)
        {
            {
                parameter::data pitchParam("Pitch (semitones)", { -48.0, 24.0, 0.01 });
                registerCallback<0>(pitchParam);
                pitchParam.setDefaultValue(0.0);
                data.add(std::move(pitchParam));
            }
            {
                parameter::data startParam("Playhead Start", { 0.0, 1.0, 0.001 });
                registerCallback<1>(startParam);
                startParam.setDefaultValue(0.0);
                data.add(std::move(startParam));
            }
            {
                parameter::data loopStartParam("Loop Start", { 0.0, 1.0, 0.001 });
                registerCallback<2>(loopStartParam);
                loopStartParam.setDefaultValue(0.0);
                data.add(std::move(loopStartParam));
            }
            {
                parameter::data loopEndParam("Sample End", { 0.0, 1.0, 0.001 });
                registerCallback<3>(loopEndParam);
                loopEndParam.setDefaultValue(1.0);
                data.add(std::move(loopEndParam));
            }
            {
                parameter::data xfadeParam("Xfade Length", { 0.0, 1.0, 0.001 });
                registerCallback<4>(xfadeParam);
                xfadeParam.setDefaultValue(0.0);
                data.add(std::move(xfadeParam));
            }
            {
                parameter::data loopModeParam("Loop Mode", { 0.0, 1.0, 1.0 });
                registerCallback<5>(loopModeParam);
                loopModeParam.setDefaultValue(0.0);
                data.add(std::move(loopModeParam));
            }
        }
    };

} // namespace project
