#include "text_writer.h"

#include <sstream>
#include <regex>
#include <iostream>
namespace nmtSample
{
    TextWriter::TextWriter(std::shared_ptr<std::ostream> textOnput, Vocabulary::ptr vocabulary)
        : mOutput(textOnput)
        , mVocabulary(vocabulary)
    {
    }

    void TextWriter::write(
        int sampleCount,
        int maxOutputSequenceLength,
        const int * hOutputData,
        const int * hActualOutputSequenceLengths)
    {
        // if clean and handle BPE outputs is required
        for(int sampleId = 0; sampleId < sampleCount; ++sampleId)
        {
            int sequenceLength = hActualOutputSequenceLengths[sampleId];
            auto currentOutputData = hOutputData + sampleId * maxOutputSequenceLength;
            *mOutput << DataWriter::generateText(sequenceLength, currentOutputData, mVocabulary) << "\n";
        }
    }

    void TextWriter::initialize()
    {
    }

    void TextWriter::finalize()
    {
    }

    std::string TextWriter::getInfo()
    {
        std::stringstream ss;
        ss << "Text Writer, vocabulary size = " << mVocabulary->getSize();
        return ss.str();
    }
}
