#include "search_policy.h"

namespace nmtSample
{
    SearchPolicy::SearchPolicy(int endSequenceId, LikelihoodCombinationOperator::ptr likelihoodCombinationOperator)
        : mEndSequenceId(endSequenceId)
        , mLikelihoodCombinationOperator(likelihoodCombinationOperator)
    {
    }
}
