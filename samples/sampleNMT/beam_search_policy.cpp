#include "beam_search_policy.h"

#include <algorithm>
#include <limits>
#include <cassert>
#include <sstream>

namespace nmtSample
{
    BeamSearchPolicy::BeamSearchPolicy(
        int endSequenceId,
        LikelihoodCombinationOperator::ptr likelihoodCombinationOperator,
        int beamWidth)
        : SearchPolicy(endSequenceId, likelihoodCombinationOperator)
        , mBeamWidth(beamWidth)
    {
    }

    void BeamSearchPolicy::initialize(
        int sampleCount,
        int * maxOutputSequenceLengths)
    {
        mSampleCount = sampleCount;
        mMaxOutputSequenceLengths.resize(mSampleCount);
        std::copy(maxOutputSequenceLengths, maxOutputSequenceLengths + mSampleCount, &mMaxOutputSequenceLengths[0]);

        // Only single ray is valid in the beam in the very beginning of beam search
        mValidRays.resize(mSampleCount * mBeamWidth);
        for(int sampleId = 0; sampleId < mSampleCount; ++sampleId)
        {
            auto currentValidRays = mValidRays.begin() + sampleId * mBeamWidth;
            *currentValidRays = true;
            std::fill(currentValidRays + 1, currentValidRays + mBeamWidth, false);
        }

        mCurrentLikelihoods.resize(mSampleCount * mBeamWidth);
        std::fill(mCurrentLikelihoods.begin(), mCurrentLikelihoods.end(), mLikelihoodCombinationOperator->init());

        mBeamSearchTable.clear();

        mTimestepId = 0;

        mCandidates.resize(mSampleCount);
        mCandidateLikelihoods.resize(mSampleCount);
        std::fill(mCandidateLikelihoods.begin(), mCandidateLikelihoods.end(), -std::numeric_limits<float>::max());
    }

    bool operator < (const BeamSearchPolicy::Option& i, const BeamSearchPolicy::Option& j)
    {
        if (i.likelihood != j.likelihood)
            return i.likelihood > j.likelihood;
        // Make comparison a total order.
        if (i.originalRayId != j.originalRayId)
            return i.originalRayId < j.originalRayId;
        return i.optionInRayId < j.optionInRayId;
    }

    void BeamSearchPolicy::processTimestep(
        int sampleCount,
        const float * hLikelihoods,
        const int * hVocabularyIndices,
        int * hSourceRayIndices,
        int * hSourceRayOptionIndices)
    {
        ++mTimestepId;
        mBeamSearchTable.resize(mTimestepId * mSampleCount * mBeamWidth);
        auto baseBeamSearchTable = mBeamSearchTable.begin() + (mTimestepId - 1) * mSampleCount * mBeamWidth;

        std::vector<Option> options;
        for(int sampleId = 0; sampleId < mSampleCount; ++sampleId)
        {
            options.clear();
            auto currentValidRays = mValidRays.begin() + sampleId * mBeamWidth;
            auto currentLikelihoods = mCurrentLikelihoods.begin() + sampleId * mBeamWidth;
            for(int rayId = 0; rayId < mBeamWidth; ++rayId)
            {
                if (*(currentValidRays + rayId))
                {
                    float rayLikelihood = *(currentLikelihoods + rayId);
                    auto currentLikelihoods = hLikelihoods + (sampleId * mBeamWidth + rayId) * mBeamWidth;
                    auto currentVocabularyIndices = hVocabularyIndices + (sampleId * mBeamWidth + rayId) * mBeamWidth;
                    for(int optionId = 0; optionId < mBeamWidth; ++optionId)
                    {
                        float optionLikelihood = *(currentLikelihoods + optionId);
                        float combinedLikelihood = mLikelihoodCombinationOperator->combine(rayLikelihood, optionLikelihood);
                        if (combinedLikelihood > mCandidateLikelihoods[sampleId]) // We drop options which are already worse (or equal to) than the finished sequence we got so far
                        {
                            int vocabularyIndex = *(currentVocabularyIndices + optionId);
                            // We consider the sequence finished if it reach the maximum length 
                            if ((vocabularyIndex == mEndSequenceId) || (mTimestepId >= mMaxOutputSequenceLengths[sampleId]))
                            {
                                // We have a new candidate output sequence for the sample
                                mCandidateLikelihoods[sampleId] = combinedLikelihood;
                                auto& candidate = mCandidates[sampleId];
                                candidate.resize(mTimestepId);
                                backtrack(mTimestepId - 2, sampleId, rayId, &candidate[0], mTimestepId - 2);
                                candidate[mTimestepId - 1] = vocabularyIndex;
                            }
                            else
                            {
                                // We have a new option with likelihood higher than the current candidate
                                Option newOption{vocabularyIndex, rayId, optionId, combinedLikelihood};
                                options.push_back(newOption);
                            }
                        }
                    }
                }
            }
            // options vector is now filled with all valid options for the sample at current timestep, it could be anything from 0 to mBeamWidth^2

            // Have top-K elements in the beginning of array            
            std::partial_sort(options.begin(), options.begin() + std::min(mBeamWidth, (int)options.size()), options.end());

            auto sourceRayIndices = hSourceRayIndices + sampleId * mBeamWidth;
            auto sourceRayOptionIndices = hSourceRayOptionIndices + sampleId * mBeamWidth;
            auto currentBeamSearchTable = baseBeamSearchTable + sampleId * mBeamWidth;
            int newRayId = 0;

            if (mTimestepId < mMaxOutputSequenceLengths[sampleId])
            {
                // Fill new beam with valid Top-K options
                for(; newRayId < std::min(mBeamWidth, (int)options.size()); ++newRayId)
                {
                    const auto& option = options[newRayId];

                    // Check if the current candidate is already better than this option
                    if (option.likelihood <= mCandidateLikelihoods[sampleId])
                        break; // The remaining options are even worse

                    *(sourceRayIndices + newRayId) = option.originalRayId;
                    *(sourceRayOptionIndices + newRayId) = option.originalRayId * mBeamWidth + option.optionInRayId;
                    *(currentValidRays + newRayId) = true;
                    *(currentLikelihoods + newRayId) = option.likelihood;
                    (currentBeamSearchTable + newRayId)->vocabularyId = option.vocabularyId;
                    (currentBeamSearchTable + newRayId)->backtrackId = option.originalRayId;
                }
            }

            // Mark the remaining rays as invalid ones
            for(; newRayId < mBeamWidth; ++newRayId)
            {
                *(sourceRayIndices + newRayId) = 0;
                *(sourceRayOptionIndices + newRayId) = 0;
                *(currentValidRays + newRayId) = false;
                *(currentLikelihoods + newRayId) = -std::numeric_limits<float>::max();
                (currentBeamSearchTable + newRayId)->vocabularyId = mEndSequenceId;
                (currentBeamSearchTable + newRayId)->backtrackId = 0;
            }
        }
    }

    bool BeamSearchPolicy::haveMoreWork()
    {
        return std::any_of(mValidRays.begin(), mValidRays.end(), [] (bool b) { return b; });
    }

    void BeamSearchPolicy::readGeneratedResult(
        int sampleCount,
        int maxOutputSequenceLength,
        int * hOutputData,
        int * hActualOutputSequenceLengths)
    {
        for(int sampleId = 0; sampleId < sampleCount; ++sampleId)
        {
            if (mCandidateLikelihoods[sampleId] > -std::numeric_limits<float>::max())
            {
                // We have a candidate (finished sequence)
                std::copy_n(
                    mCandidates[sampleId].begin(),
                    std::min(static_cast<int>(mCandidates[sampleId].size()), maxOutputSequenceLength),
                    hOutputData + sampleId * maxOutputSequenceLength);
                hActualOutputSequenceLengths[sampleId] = mCandidates[sampleId].size();
            }
            else
            {
                // We don't have a finished sequence generated, will output the unfinished one with the highest likelihood
                assert(mValidRays[sampleId * mBeamWidth]);
                backtrack(mTimestepId - 1, sampleId, 0, hOutputData + sampleId * maxOutputSequenceLength, maxOutputSequenceLength - 1);
                hActualOutputSequenceLengths[sampleId] = mTimestepId;
            }
        }
    }

    void BeamSearchPolicy::backtrack(
        int lastTimestepId,
        int sampleId,
        int lastTimestepRayId,
        int * hOutputData,
        int lastTimestepWriteId) const
    {
        int rayId = lastTimestepRayId;
        for(int timestepId = lastTimestepId; timestepId >= 0; --timestepId)
        {
            const auto& entry = mBeamSearchTable[(timestepId * mSampleCount + sampleId) * mBeamWidth + rayId];
            rayId = entry.backtrackId;
            if (timestepId <= lastTimestepWriteId)
                hOutputData[timestepId] = entry.vocabularyId;
        }
    }

    std::string BeamSearchPolicy::getInfo()
    {
        std::stringstream ss;
        ss << "Beam Search Policy, beam = " << mBeamWidth;
        return ss.str();
    }
}
