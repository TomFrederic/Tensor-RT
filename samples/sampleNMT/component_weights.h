#ifndef SAMPLE_NMT_COMPONENT_WEIGHTS_
#define SAMPLE_NMT_COMPONENT_WEIGHTS_

#include <memory>
#include <iostream>
#include <vector>

namespace nmtSample
{
    /** \class ComponentWeights
    *
    * \brief weights storage 
    *
    */
    class ComponentWeights
    {
    public:
        typedef std::shared_ptr<ComponentWeights> ptr;

        ComponentWeights() = default;

        virtual ~ComponentWeights() = default;

        friend std::istream& operator>>(std::istream& input, ComponentWeights& value);

    public:
        std::vector<int> mMetaData;
        std::vector<char> mWeights;
    };
}

#endif // SAMPLE_NMT_COMPONENT_WEIGHTS_
