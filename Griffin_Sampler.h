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

    // Sample settings now include playbackStart, loopStart and loopEnd.
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

    // SamplePlayback: stores playback pointers and loop boundaries.
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
            if (pbStart < 0.f) pbStart = 0.f;
            if (pbStart > float(bufferLength - 1)) pbStart = float(bufferLength - 1);
            playbackStart = pbStart;

            float lStart = s.loopStartInSamples;
            float lEnd = s.loopEndInSamples;
            if (!s.loopMode)
            {
                lStart = pbStart;
                lEnd = s.loopEndInSamples; // one-shot: use parameter as sample end
            }
            else
            {
                if (lStart < 0.f) lStart = 0.f;
                if (lStart > float(bufferLength - 1)) lStart = float(bufferLength - 1);
                if (lEnd < 0.f) lEnd = 0.f;
                if (lEnd > float(bufferLength - 1)) lEnd = float(bufferLength - 1);
            }
            loopStart = lStart;
            loopEnd = lEnd;

            phaseAcc = (int64_t)std::llround(pbStart * FIXED_ONE);
            double centsFact = std::pow(2.0, (s.pitchOffsetCents / 1200.0));
            double effectiveSpeed = baseDelta * centsFact;
            phaseInc = (int64_t)std::llround(effectiveSpeed * FIXED_ONE);
        }

        // New method: update settings for an active voice without resetting phaseAcc.
        FORCE_INLINE void updateSettings(const SampleSettings& s, int bufferLength)
        {
            settings = s;
            float pbStart = s.playbackStartInSamples;
            if (pbStart < 0.f) pbStart = 0.f;
            if (pbStart > float(bufferLength - 1)) pbStart = float(bufferLength - 1);
            // Do not modify phaseAcc to preserve current playback position.
            playbackStart = pbStart;
            float lStart = s.loopStartInSamples;
            float lEnd = s.loopEndInSamples;
            if (!s.loopMode)
            {
                lStart = pbStart;
                lEnd = s.loopEndInSamples; // one-shot: use provided loop end parameter
            }
            else
            {
                if (lStart < 0.f) lStart = 0.f;
                if (lStart > float(bufferLength - 1)) lStart = float(bufferLength - 1);
                if (lEnd < 0.f) lEnd = 0.f;
                if (lEnd > float(bufferLength - 1)) lEnd = float(bufferLength - 1);
            }
            loopStart = lStart;
            loopEnd = lEnd;
        }

        // vectorSynthesize outputs one block of samples.
        // For looping: the unique region is [loopStart, loopEnd - X) and the crossfade zone is [loopEnd - X, loopEnd).
        // When p reaches loopEnd, we subtract (loopEnd - (loopStart + X)) so that if p == loopEnd, new p == loopStart + X.
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
            const float X = settings.xfadeLengthInSamples; // crossfade length
            const float endSample = loopEnd; // B
            // Crossfade zone from (B - X) to B.
            const float crossfadeStart = (loopEnabled ? endSample - X : 0.f);
            const float piOverTwo = float(M_PI * 0.5f);

            while (processed < blockSize)
            {
                float p = float(phaseAcc >> FIXED_SHIFT) +
                    float(phaseAcc & FIXED_MASK) * invFixedOne;

                if (!loopEnabled)
                {
                    if (p >= endSample)
                    {
                        active = false;
                        break;
                    }
                }
                else
                {
                    if (p < loopStart)
                    {
                        // Not yet in loop region.
                    }
                    else
                    {
                        if (p >= endSample)
                        {
                            // Compute excess beyond loopEnd.
                            float excess = p - endSample;
                            // Effective loop length is (endSample - (loopStart + X)).
                            p = loopStart + X + excess;
                            phaseAcc = (int64_t)std::llround(p * FIXED_ONE);
                        }
                        else if (p >= crossfadeStart)
                        {
                            // In crossfade zone: output from precomputed buffer if available.
                            if (preXfadeBuffer != nullptr)
                            {
                                float posInXfade = p - crossfadeStart; // 0..X
                                int idx = int(posInXfade);
                                float frac = posInXfade - idx;
                                int bufferLength = preXfadeBuffer->getNumSamples();
                                if (idx < 0)
                                    idx = 0;
                                else if (idx >= bufferLength - 1)
                                    idx = bufferLength - 2;
                                const float* xfadeL = preXfadeBuffer->getReadPointer(0);
                                const float* xfadeR = preXfadeBuffer->getReadPointer(1);
                                float sampL = xfadeL[idx] + frac * (xfadeL[idx + 1] - xfadeL[idx]);
                                float sampR = xfadeR[idx] + frac * (xfadeR[idx + 1] - xfadeR[idx]);
                                outL[processed] += sampL * ampLeft;
                                outR[processed] += sampR * ampRight;
                                processed++;
                                phaseAcc += phaseInc;
                                continue;
                            }
                            else
                            {
                                // Fallback on-the-fly computation.
                                float alpha = (p - crossfadeStart) / X;
                                float crossAngle = alpha * piOverTwo;
                                float tailGain = std::cos(crossAngle);
                                float headGain = std::sin(crossAngle);
                                int idx = int(p);
                                float frac = p - idx;
                                float sampTailL = sourceL[idx] + frac * (sourceL[idx + 1] - sourceL[idx]);
                                float sampTailR = sourceR[idx] + frac * (sourceR[idx + 1] - sourceR[idx]);
                                float headPos = loopStart + (p - crossfadeStart);
                                int idxHead = int(headPos);
                                float fracHead = headPos - idxHead;
                                float sampHeadL = sourceL[idxHead] + fracHead * (sourceL[idxHead + 1] - sourceL[idxHead]);
                                float sampHeadR = sourceR[idxHead] + fracHead * (sourceR[idxHead + 1] - sourceR[idxHead]);
                                sampTailL = tailGain * sampTailL + headGain * sampHeadL;
                                sampTailR = tailGain * sampTailR + headGain * sampHeadR;
                                outL[processed] += sampTailL * ampLeft;
                                outR[processed] += sampTailR * ampRight;
                                processed++;
                                phaseAcc += phaseInc;
                                continue;
                            }
                        }
                    }
                }

                // Normal playback (outside loop region or before crossfade)
                int idx = int(p);
                float frac = p - float(idx);
                float sampL = sourceL[idx] + frac * (sourceL[idx + 1] - sourceL[idx]);
                float sampR = sourceR[idx] + frac * (sourceR[idx + 1] - sourceR[idx]);

                outL[processed] += sampL * ampLeft;
                outR[processed] += sampR * ampRight;
                processed++;
                phaseAcc += phaseInc;
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

    // Griffin_Sampler now uses six parameters.
    // Parameter 1: Playhead Start percent
    // Parameter 2: Loop Start percent
    // Parameter 3: Loop End percent
    // Parameter 4: Xfade Length (relative fraction)
    // Parameter 5: Loop Mode (enabled if >= 0.5)
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
                if (xfadeFraction < 0.f) xfadeFraction = 0.f;
                if (xfadeFraction > 1.f) xfadeFraction = 1.f;
                updateDerivedParameters();
            }
            else if constexpr (P == 5)
            {
                loopMode = (v >= 0.5);
                updateDerivedParameters();
            }
            if constexpr (P >= 1 && P <= 5)
            {
                // Update all active voices with the new boundaries and parameters.
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
                parameter::data pitchParam("Pitch (semitones)", { -24.0, 24.0, 0.01 });
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
                parameter::data loopEndParam("Loop End", { 0.0, 1.0, 0.001 });
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
