#include <iostream>
#include <exception>
#include <utility>
#include <vector>
#include <fstream>
#include <cstring>
#include <cuda_runtime.h>
#include <sstream>
#include <algorithm>

#include "device_buffer.h"
#include "pinned_host_buffer.h"
#include "sequence_properties.h"
#include "data_reader.h"
#include "vocabulary.h"
#include "text_reader.h"
#include "embedder.h"
#include "encoder.h"
#include "decoder.h"
#include "context.h"
#include "attention.h"
#include "projection.h"
#include "likelihood.h"
#include "search_policy.h"
#include "data_writer.h"
#include "trt_builder.h"
#include "trt_network.h"
#include "trt_engine.h"
#include "trt_context.h"
#include "component_weights.h"
#include "alignment.h"
#include "debug_util.h"
#include "trt_util.h"
#include "limited_samples_data_reader.h"

#include "slp_embedder.h"
#include "softmax_likelihood.h"
#include "lstm_encoder.h"
#include "lstm_decoder.h"
#include "multiplicative_alignment.h"
#include "slp_attention.h"
#include "slp_projection.h"
#include "beam_search_policy.h"
#include "bleu_score_writer.h"
#include "text_writer.h"
#include "benchmark_writer.h"

#include "NvInfer.h"

bool gPrintComponentInfo = true;
bool gFeedAttentionToInput = true;
bool gInitializeDecoderFromEncoderHiddenStates = true;

int gMaxBatchSize = 128;
int gBeamWidth = 5;
int gMaxInputSequenceLength = 150;
int gMaxOutputSequenceLength = -1;
int gMaxInferenceSamples = -1;
std::string gDataWriterStr = "bleu";
std::string gOutputTextFileName("translation_output.txt");
bool gVerbose = false;
int gMaxWorkspaceSize = (1 << 28);
std::string gDataDirectory("../../../../data/samples/nmt/deen");

std::string gInputTextFileName("newstest2015.tok.bpe.32000.de");
std::string gReferenceOutputTextFileName("newstest2015.tok.bpe.32000.en");
std::string gInputVocabularyFileName("vocab.bpe.32000.de"); 
std::string gOutputVocabularyFileName("vocab.bpe.32000.en"); 
std::string gEncEmbedFileName("weights/encembed.bin");
std::string gEncRnnFileName("weights/encrnn.bin");
std::string gDecEmbedFileName("weights/decembed.bin");
std::string gDecRnnFileName("weights/decrnn.bin");
std::string gDecAttFileName("weights/decatt.bin");
std::string gDecMemFileName("weights/decmem.bin");
std::string gDecProjFileName("weights/decproj.bin");
nmtSample::Vocabulary::ptr gOutputVocabulary = std::make_shared<nmtSample::Vocabulary>();

nmtSample::SequenceProperties::ptr getOutputSequenceProperties()
{
    return gOutputVocabulary;
}

nmtSample::DataReader::ptr getDataReader()
{
    std::shared_ptr<std::istream> textInput(new std::ifstream(gDataDirectory + "/" + gInputTextFileName));
    std::shared_ptr<std::istream> vocabInput(new std::ifstream(gDataDirectory + "/" + gInputVocabularyFileName));
    assert(textInput->good());
    assert(vocabInput->good());

    auto vocabulary = std::make_shared<nmtSample::Vocabulary>();
    *vocabInput >> *vocabulary;

    auto reader = std::make_shared<nmtSample::TextReader>(textInput, vocabulary);

    if (gMaxInferenceSamples >= 0)
        return std::make_shared<nmtSample::LimitedSamplesDataReader>(gMaxInferenceSamples, reader);
    else
        return reader;
}

nmtSample::Embedder::ptr getInputEmbedder()
{
    auto weights = std::make_shared<nmtSample::ComponentWeights>();
    
    std::ifstream input(gDataDirectory + "/" + gEncEmbedFileName);
    assert(input.good());
    input >> *weights; 
    
    return std::make_shared<nmtSample::SLPEmbedder>(weights);
}

nmtSample::Embedder::ptr getOutputEmbedder()
{
    auto weights = std::make_shared<nmtSample::ComponentWeights>();
    
    std::ifstream input(gDataDirectory + "/" + gDecEmbedFileName);
    assert(input.good());
    input >> *weights; 

    return std::make_shared<nmtSample::SLPEmbedder>(weights);
}

nmtSample::Encoder::ptr getEncoder()
{
    auto weights = std::make_shared<nmtSample::ComponentWeights>();

    std::ifstream input(gDataDirectory + "/" + gEncRnnFileName);
    assert(input.good());
    input >> *weights; 

    return std::make_shared<nmtSample::LSTMEncoder>(weights);
}

nmtSample::Alignment::ptr getAlignment()
{
    auto weights = std::make_shared<nmtSample::ComponentWeights>();

    std::ifstream input(gDataDirectory + "/" + gDecMemFileName);
    assert(input.good());
    input >> *weights; 

    return std::make_shared<nmtSample::MultiplicativeAlignment>(weights);
}

nmtSample::Context::ptr getContext()
{
    return std::make_shared<nmtSample::Context>();
}

nmtSample::Decoder::ptr getDecoder()
{
    auto weights = std::make_shared<nmtSample::ComponentWeights>();

    std::ifstream input(gDataDirectory + "/" + gDecRnnFileName);
    assert(input.good());
    input >> *weights;

    return std::make_shared<nmtSample::LSTMDecoder>(weights);
}

nmtSample::Attention::ptr getAttention()
{
    auto weights = std::make_shared<nmtSample::ComponentWeights>();

    std::ifstream input(gDataDirectory + "/" + gDecAttFileName);
    assert(input.good());
    input >> *weights;

    return std::make_shared<nmtSample::SLPAttention>(weights);
}

nmtSample::Projection::ptr getProjection()
{
    auto weights = std::make_shared<nmtSample::ComponentWeights>();

    std::ifstream input(gDataDirectory + "/" + gDecProjFileName);
    assert(input.good());
    input >> *weights;

    return std::make_shared<nmtSample::SLPProjection>(weights);
}

nmtSample::Likelihood::ptr getLikelihood()
{
    return std::make_shared<nmtSample::SoftmaxLikelihood>();
}

nmtSample::SearchPolicy::ptr getSearchPolicy(int endSequenceId, nmtSample::LikelihoodCombinationOperator::ptr likelihoodCombinationOperator)
{
    return std::make_shared<nmtSample::BeamSearchPolicy>(endSequenceId, likelihoodCombinationOperator, gBeamWidth);
}

nmtSample::DataWriter::ptr getDataWriter()
{
    if (gDataWriterStr == "bleu")
    {
        std::shared_ptr<std::istream> textInput(new std::ifstream(gDataDirectory + "/" + gReferenceOutputTextFileName));
        assert(textInput->good());
        return std::make_shared<nmtSample::BLEUScoreWriter>(textInput, gOutputVocabulary);
    }
    else if (gDataWriterStr == "text")
    {
        std::shared_ptr<std::ostream> textOutput(new std::ofstream(gOutputTextFileName));
        assert(textOutput->good());
        return std::make_shared<nmtSample::TextWriter>(textOutput, gOutputVocabulary);
    }
    else if (gDataWriterStr == "benchmark")
    {
        return std::make_shared<nmtSample::BenchmarkWriter>();
    }
    else
    {
        std::cerr << "Invalid data writer specified: " << gDataWriterStr << std::endl;
        assert(0);
        return nmtSample::DataWriter::ptr();
    }
}

bool parseString(const char * arg, const char * name, std::string& value)
{
	size_t n = strlen(name);
	bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
	if (match)
	{
		value = arg + n + 3;
		std::cout << name << ": " << value << std::endl;
	}
	return match;
}

bool parseInt(const char * arg, const char * name, int& value)
{
	size_t n = strlen(name);
	bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
	if (match)
	{
		value = atoi(arg + n + 3);
		std::cout << name << ": " << value << std::endl;
	}
	return match;
}

bool parseBool(const char * arg, const char * name, bool& value)
{
	size_t n = strlen(name);
	bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n);
	if (match)
	{
		std::cout << name << ": true" << std::endl;
		value = true;
	}
	return match;
}

void printUsage()
{
	printf("\nOptional params:\n");
	printf("  --help                                   Output help message and exit\n");
	printf("  --data_writer=bleu/text/benchmark        Type of the output the app generates (default = %s)\n", gDataWriterStr.c_str());
	printf("  --output_file=<path_to_file>             Path to the output file when data_writer=text (default = %s)\n", gOutputTextFileName.c_str());
	printf("  --batch=<N>                              Batch size (default = %d)\n", gMaxBatchSize);
	printf("  --beam=<N>                               Beam width (default = %d)\n", gBeamWidth);
	printf("  --max_input_sequence_length=<N>          Maximum length for input sequences (default = %d)\n", gMaxInputSequenceLength);
	printf("  --max_output_sequence_length=<N>         Maximum length for output sequences (default = %d), negative value indicates no limit\n", gMaxOutputSequenceLength);
	printf("  --max_inference_samples=<N>              Maximum sample count to run inference for, negative values indicates no limit is set (default = %d)\n", gMaxInferenceSamples);
	printf("  --verbose                                Output info-level messages by TensorRT\n");
	printf("  --max_workspace_size=<N>                 Maximum workspace size (default = %d)\n", gMaxWorkspaceSize);
	printf("  --data_dir=<path_to_data_directory>      Path to the directory where data and weights are located (default = %s)\n", gDataDirectory.c_str());
}

bool parseArgs(int argc, char* argv[])
{
	if (argc < 1)
	{
		printUsage();
		return false;
	}

    bool showHelp = false;
	for (int j = 1; j < argc; j++)
    {
        if (parseBool(argv[j], "help", showHelp))
            continue;
        if (parseString(argv[j], "data_writer", gDataWriterStr))
            continue;
        if (parseString(argv[j], "output_file", gOutputTextFileName))
            continue;
        if (parseInt(argv[j], "batch", gMaxBatchSize))
            continue;
        if (parseInt(argv[j], "beam", gBeamWidth))
            continue;
        if (parseInt(argv[j], "max_input_sequence_length", gMaxInputSequenceLength))
            continue;
        if (parseInt(argv[j], "max_output_sequence_length", gMaxOutputSequenceLength))
            continue;
        if (parseInt(argv[j], "max_inference_samples", gMaxInferenceSamples))
            continue;
        if (parseBool(argv[j], "verbose", gVerbose))
            continue;
        if (parseInt(argv[j], "max_workspace_size", gMaxWorkspaceSize))
            continue;
        if (parseString(argv[j], "data_dir", gDataDirectory))
            continue;
    }

    if (showHelp)
    {
        printUsage();
        return false;
    }

    return true;
}

nmtSample::TRTEngine::ptr getEncoderEngine(
    nmtSample::Embedder::ptr inputEmbedder,
    nmtSample::Encoder::ptr encoder,
    nmtSample::Alignment::ptr alignment)
{
    nmtSample::TRTBuilder encoderBuilder(gMaxBatchSize, gMaxWorkspaceSize, gVerbose);
    nmtSample::TRTNetwork encoderNetwork(encoderBuilder.getImpl()->createNetwork());

    // Define inputs for the encoder
    nvinfer1::Dims inputDims{1, {gMaxInputSequenceLength}, {nvinfer1::DimensionType::kSEQUENCE}};
    auto inputEncoderDataTensor = encoderNetwork.getImpl()->addInput("input_encoder_data", nvinfer1::DataType::kINT32, inputDims);
    assert(inputEncoderDataTensor != nullptr);
    nvinfer1::Dims inputSequenceLengthsDims{1, {1}, {nvinfer1::DimensionType::kINDEX}};
    auto actualInputSequenceLengthsTensor = encoderNetwork.getImpl()->addInput("actual_input_sequence_lengths", nvinfer1::DataType::kINT32, inputSequenceLengthsDims);
    assert(actualInputSequenceLengthsTensor != nullptr);

    auto stateSizes = encoder->getStateSizes();
    std::vector<nvinfer1::ITensor *> encoderInputStatesTensors(stateSizes.size());
    for(int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "input_encoder_states_" << i;
        nvinfer1::Dims statesDims{1, {stateSizes[i]}, {nvinfer1::DimensionType::kCHANNEL}};
        encoderInputStatesTensors[i] = encoderNetwork.getImpl()->addInput(ss.str().c_str(), nvinfer1::DataType::kFLOAT, statesDims);
        assert(encoderInputStatesTensors[i] != nullptr);
    }

    nvinfer1::ITensor * initializeDecoderIndicesTensor = nullptr;
    if (gInitializeDecoderFromEncoderHiddenStates)
    {
        nvinfer1::Dims inputDims{1, {gBeamWidth}, {nvinfer1::DimensionType::kINDEX}};
        initializeDecoderIndicesTensor = encoderNetwork.getImpl()->addInput("initialize_decoder_indices", nvinfer1::DataType::kINT32, inputDims);
        assert(initializeDecoderIndicesTensor != nullptr);
    }

    nvinfer1::ITensor * inputEncoderEmbeddedTensor;
    inputEmbedder->addToModel(encoderNetwork.getImpl(), inputEncoderDataTensor, &inputEncoderEmbeddedTensor);
    inputEncoderEmbeddedTensor->setName("input_data_embedded");

    nvinfer1::ITensor * memoryStatesTensor;
    std::vector<nvinfer1::ITensor *> encoderOutputStatesTensors(stateSizes.size());
    encoder->addToModel(
        encoderNetwork.getImpl(),
        gMaxInputSequenceLength,
        inputEncoderEmbeddedTensor,
        actualInputSequenceLengthsTensor,
        &encoderInputStatesTensors[0],
        &memoryStatesTensor,
        gInitializeDecoderFromEncoderHiddenStates ? &encoderOutputStatesTensors[0] : nullptr);
    memoryStatesTensor->setName("memory_states");
    encoderNetwork.getImpl()->markOutput(*memoryStatesTensor);

    if (alignment->getAttentionKeySize() > 0)
    {
        nvinfer1::ITensor * attentionKeysTensor;
        alignment->addAttentionKeys(
            encoderNetwork.getImpl(),
            memoryStatesTensor,
            &attentionKeysTensor);
        attentionKeysTensor->setName("attention_keys");
        encoderNetwork.getImpl()->markOutput(*attentionKeysTensor);
    }

    // Replicate sequence lengths for the decoder
    {
        auto gatherLayer = encoderNetwork.getImpl()->addGather(*actualInputSequenceLengthsTensor, *initializeDecoderIndicesTensor, 0);
        assert(gatherLayer != nullptr);
        auto actualInputSequenceLengthsReplicatedTensor = gatherLayer->getOutput(0);
        assert(actualInputSequenceLengthsReplicatedTensor != nullptr);
        actualInputSequenceLengthsReplicatedTensor->setName("actual_input_sequence_lengths_replicated");
        encoderNetwork.getImpl()->markOutput(*actualInputSequenceLengthsReplicatedTensor);
        actualInputSequenceLengthsReplicatedTensor->setType(nvinfer1::DataType::kINT32);
    }

    if (gInitializeDecoderFromEncoderHiddenStates)
    {
        for(int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
        {
            assert(encoderOutputStatesTensors[i] != nullptr);

            // Insert index (Z=1) dimension into tensor
            nvinfer1::ITensor * encoderOutputStatesTensorWithUnitIndex;
            {
                auto shuffleLayer = encoderNetwork.getImpl()->addShuffle(*encoderOutputStatesTensors[i]);
                assert(shuffleLayer != nullptr);
                nvinfer1::Dims shuffleDims{2, {1, stateSizes[i]}, {nvinfer1::DimensionType::kINDEX, nvinfer1::DimensionType::kCHANNEL}};
                shuffleLayer->setReshapeDimensions(shuffleDims);
                encoderOutputStatesTensorWithUnitIndex = shuffleLayer->getOutput(0);
                assert(encoderOutputStatesTensorWithUnitIndex != nullptr);
            }
            auto gatherLayer = encoderNetwork.getImpl()->addGather(*encoderOutputStatesTensorWithUnitIndex, *initializeDecoderIndicesTensor, 0);
            assert(gatherLayer != nullptr);
            auto inputDecoderHiddenStatesTensor = gatherLayer->getOutput(0);
            assert(inputDecoderHiddenStatesTensor != nullptr);
            std::stringstream ss;
            ss << "input_decoder_states_" << i;
            inputDecoderHiddenStatesTensor->setName(ss.str().c_str());
            encoderNetwork.getImpl()->markOutput(*inputDecoderHiddenStatesTensor);
        }
    }

    return std::make_shared<nmtSample::TRTEngine>(encoderBuilder.getImpl()->buildCudaEngine(*encoderNetwork.getImpl()));
}

nmtSample::TRTEngine::ptr getGeneratorEngine(
    nmtSample::Embedder::ptr outputEmbedder,
    nmtSample::Decoder::ptr decoder,
    nmtSample::Alignment::ptr alignment,
    nmtSample::Context::ptr context,
    nmtSample::Attention::ptr attention,
    nmtSample::Projection::ptr projection,
    nmtSample::Likelihood::ptr likelihood)
{
    nmtSample::TRTBuilder generatorBuilder(gMaxBatchSize, gMaxWorkspaceSize, gVerbose);
    nmtSample::TRTNetwork generatorNetwork(generatorBuilder.getImpl()->createNetwork());

    // Define inputs for the generator
    auto stateSizes = decoder->getStateSizes();
    std::vector<nvinfer1::ITensor *> decoderInputStatesTensors(stateSizes.size());
    for(int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "input_decoder_states_" << i;
        nvinfer1::Dims statesDims{2, {gBeamWidth, stateSizes[i]}, {nvinfer1::DimensionType::kINDEX, nvinfer1::DimensionType::kCHANNEL}};
        decoderInputStatesTensors[i] = generatorNetwork.getImpl()->addInput(ss.str().c_str(), nvinfer1::DataType::kFLOAT, statesDims);
        assert(decoderInputStatesTensors[i] != nullptr);
    }
    nvinfer1::Dims inputDecoderDataDims{1, {gBeamWidth}, {nvinfer1::DimensionType::kINDEX}};
    auto inputDecoderDataTensor = generatorNetwork.getImpl()->addInput("input_decoder_data", nvinfer1::DataType::kINT32, inputDecoderDataDims);
    assert(inputDecoderDataTensor != nullptr);
    nvinfer1::Dims inputSequenceLengthsTeplicatedDims{2, {gBeamWidth, 1}, {nvinfer1::DimensionType::kINDEX, nvinfer1::DimensionType::kCHANNEL}};
    auto actualInputSequenceLengthsReplicatedTensor = generatorNetwork.getImpl()->addInput("actual_input_sequence_lengths_replicated", nvinfer1::DataType::kINT32, inputSequenceLengthsTeplicatedDims);
    assert(actualInputSequenceLengthsReplicatedTensor != nullptr);
    nvinfer1::Dims memoryStatesDims{2, {gMaxInputSequenceLength, alignment->getSourceStatesSize()}, {nvinfer1::DimensionType::kSEQUENCE, nvinfer1::DimensionType::kCHANNEL}};
    auto memoryStatesTensor = generatorNetwork.getImpl()->addInput("memory_states", nvinfer1::DataType::kFLOAT, memoryStatesDims);
    assert(memoryStatesTensor != nullptr);
    nvinfer1::ITensor * attentionKeysTensor = nullptr;
    if (alignment->getAttentionKeySize() > 0)
    {
        nvinfer1::Dims attentionKeysDims{2, {gMaxInputSequenceLength, alignment->getAttentionKeySize()}, {nvinfer1::DimensionType::kSEQUENCE, nvinfer1::DimensionType::kCHANNEL}};
        attentionKeysTensor = generatorNetwork.getImpl()->addInput("attention_keys", nvinfer1::DataType::kFLOAT, attentionKeysDims);
        assert(attentionKeysTensor != nullptr);
    }
    nvinfer1::ITensor * inputAttentionTensor = nullptr;
    if (gFeedAttentionToInput)
    {
        nvinfer1::Dims inputAttentionDims{2, {gBeamWidth, attention->getAttentionSize()}, {nvinfer1::DimensionType::kINDEX, nvinfer1::DimensionType::kCHANNEL}};
        inputAttentionTensor = generatorNetwork.getImpl()->addInput("input_attention", nvinfer1::DataType::kFLOAT, inputAttentionDims);
        assert(inputAttentionTensor != nullptr);
    }

    // Add output embedder
    nvinfer1::ITensor * inputDecoderEmbeddedTensor;
    outputEmbedder->addToModel(generatorNetwork.getImpl(), inputDecoderDataTensor, &inputDecoderEmbeddedTensor);
    assert(inputDecoderEmbeddedTensor != nullptr);

    // Add concatination of previous attention vector and embedded input for the decoder
    nvinfer1::ITensor * inputDecoderEmbeddedConcatinatedWithAttentionTensor;
    if (gFeedAttentionToInput)
    {
        nvinfer1::ITensor * inputTensors[] = {inputDecoderEmbeddedTensor, inputAttentionTensor};
        auto concatLayer = generatorNetwork.getImpl()->addConcatenation(inputTensors, 2);
        assert(concatLayer != nullptr);
        concatLayer->setAxis(1);
        inputDecoderEmbeddedConcatinatedWithAttentionTensor = concatLayer->getOutput(0);
        assert(inputDecoderEmbeddedConcatinatedWithAttentionTensor != nullptr);
    }

    // Add decoder (single timestep)
    nvinfer1::ITensor * outputDecoderDataTensor;
    std::vector<nvinfer1::ITensor *> decoderOutputStatesTensors(stateSizes.size());
    decoder->addToModel(generatorNetwork.getImpl(), gFeedAttentionToInput ? inputDecoderEmbeddedConcatinatedWithAttentionTensor : inputDecoderEmbeddedTensor,
        &decoderInputStatesTensors[0], &outputDecoderDataTensor, &decoderOutputStatesTensors[0]);
    for(int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "output_decoder_states_" << i;
        decoderOutputStatesTensors[i]->setName(ss.str().c_str());
        generatorNetwork.getImpl()->markOutput(*decoderOutputStatesTensors[i]);
    }

    // Add alignment scores
    nvinfer1::ITensor * alignmentScoresTensor;
    alignment->addToModel(
        generatorNetwork.getImpl(),
        (alignment->getAttentionKeySize() > 0) ? attentionKeysTensor : memoryStatesTensor,
        outputDecoderDataTensor,
        &alignmentScoresTensor);

    // Add context
    nvinfer1::ITensor * contextTensor;
    context->addToModel(generatorNetwork.getImpl(), actualInputSequenceLengthsReplicatedTensor, memoryStatesTensor, alignmentScoresTensor, &contextTensor);

    // Add attention
    nvinfer1::ITensor * attentionTensor;
    attention->addToModel(generatorNetwork.getImpl(), outputDecoderDataTensor, contextTensor, &attentionTensor);
    if (gFeedAttentionToInput)
    {
        attentionTensor->setName("output_attention");
        generatorNetwork.getImpl()->markOutput(*attentionTensor);
    }

    // Add projection
    nvinfer1::ITensor * logitsTensor;
    projection->addToModel(generatorNetwork.getImpl(), attentionTensor, &logitsTensor);

    // Add per-ray top-k options generation
    nvinfer1::ITensor * outputLikelihoodsTensor;
    nvinfer1::ITensor * outputVocabularyIndicesTensor;
    likelihood->addToModel(generatorNetwork.getImpl(), gBeamWidth, logitsTensor, &outputLikelihoodsTensor, &outputVocabularyIndicesTensor);
    outputLikelihoodsTensor->setName("output_likelihoods");
    generatorNetwork.getImpl()->markOutput(*outputLikelihoodsTensor);
    outputVocabularyIndicesTensor->setName("output_vocabulary_indices");
    generatorNetwork.getImpl()->markOutput(*outputVocabularyIndicesTensor);
    outputVocabularyIndicesTensor->setType(nvinfer1::DataType::kINT32);

    return std::make_shared<nmtSample::TRTEngine>(generatorBuilder.getImpl()->buildCudaEngine(*generatorNetwork.getImpl()));
}

nmtSample::TRTEngine::ptr getGeneratorShuffleEngine(const std::vector<int>& decoderStateSizes, int attentionSize)
{
    nmtSample::TRTBuilder shuffleBuilder(gMaxBatchSize, gMaxWorkspaceSize, gVerbose);
    nmtSample::TRTNetwork shuffleNetwork(shuffleBuilder.getImpl()->createNetwork());

    nvinfer1::Dims sourceRayIndicesDims{1, {gBeamWidth}, {nvinfer1::DimensionType::kINDEX}};
    auto sourceRayIndicesTensor = shuffleNetwork.getImpl()->addInput("source_ray_indices", nvinfer1::DataType::kINT32, sourceRayIndicesDims);
    assert(sourceRayIndicesTensor != nullptr);
    nvinfer1::Dims sourceRayOptionIndicesDims{1, {gBeamWidth}, {nvinfer1::DimensionType::kINDEX}};
    auto sourceRayOptionIndicesTensor = shuffleNetwork.getImpl()->addInput("source_ray_option_indices", nvinfer1::DataType::kINT32, sourceRayOptionIndicesDims);
    assert(sourceRayOptionIndicesTensor != nullptr);
    nvinfer1::Dims previousOutputVocabularyIndicesDims{1, {gBeamWidth * gBeamWidth}, {nvinfer1::DimensionType::kINDEX}};
    auto previousOutputVocabularyIndicesTensor = shuffleNetwork.getImpl()->addInput("previous_output_vocabulary_indices", nvinfer1::DataType::kINT32, previousOutputVocabularyIndicesDims);
    assert(previousOutputVocabularyIndicesTensor != nullptr);

    std::vector<nvinfer1::ITensor *> previousOutputDecoderStatesTensors(decoderStateSizes.size());
    for(int i = 0; i < static_cast<int>(decoderStateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "previous_output_decoder_states_" << i;
        nvinfer1::Dims statesDims{2, {gBeamWidth, decoderStateSizes[i]}, {nvinfer1::DimensionType::kINDEX, nvinfer1::DimensionType::kCHANNEL}};
        previousOutputDecoderStatesTensors[i] = shuffleNetwork.getImpl()->addInput(ss.str().c_str(), nvinfer1::DataType::kFLOAT, statesDims);
        assert(previousOutputDecoderStatesTensors[i] != nullptr);
    }
    
    nvinfer1::ITensor * previousOutputAttentionTensor = nullptr;
    if (gFeedAttentionToInput)
    {
        nvinfer1::Dims previousOutputAttentionDims{2, {gBeamWidth, attentionSize}, {nvinfer1::DimensionType::kINDEX, nvinfer1::DimensionType::kCHANNEL}};
        previousOutputAttentionTensor = shuffleNetwork.getImpl()->addInput("previous_output_attention", nvinfer1::DataType::kFLOAT, previousOutputAttentionDims);
        assert(previousOutputAttentionTensor != nullptr);
    }

    {
        auto gatherLayer = shuffleNetwork.getImpl()->addGather(*previousOutputVocabularyIndicesTensor, *sourceRayOptionIndicesTensor, 0);
        assert(gatherLayer != nullptr);
        auto inputDecoderDataTensor = gatherLayer->getOutput(0);
        assert(inputDecoderDataTensor != nullptr);
        inputDecoderDataTensor->setName("input_decoder_data");
        shuffleNetwork.getImpl()->markOutput(*inputDecoderDataTensor);
        inputDecoderDataTensor->setType(nvinfer1::DataType::kINT32);
    }

    for(int i = 0; i < static_cast<int>(decoderStateSizes.size()); ++i)
    {
        auto gatherLayer = shuffleNetwork.getImpl()->addGather(*previousOutputDecoderStatesTensors[i], *sourceRayIndicesTensor, 0);
        assert(gatherLayer != nullptr);
        auto inputDecoderHiddenStatesTensor = gatherLayer->getOutput(0);
        assert(inputDecoderHiddenStatesTensor != nullptr);
        std::stringstream ss;
        ss << "input_decoder_states_" << i;
        inputDecoderHiddenStatesTensor->setName(ss.str().c_str());
        shuffleNetwork.getImpl()->markOutput(*inputDecoderHiddenStatesTensor);
    }

    if (gFeedAttentionToInput)
    {
        auto gatherLayer = shuffleNetwork.getImpl()->addGather(*previousOutputAttentionTensor, *sourceRayIndicesTensor, 0);
        assert(gatherLayer != nullptr);
        auto inputAttentionTensor = gatherLayer->getOutput(0);
        assert(inputAttentionTensor != nullptr);
        inputAttentionTensor->setName("input_attention");
        shuffleNetwork.getImpl()->markOutput(*inputAttentionTensor);
    }

    return std::make_shared<nmtSample::TRTEngine>(shuffleBuilder.getImpl()->buildCudaEngine(*shuffleNetwork.getImpl()));
}

int main(int argc, char** argv)
{
    if (!parseArgs(argc, argv))
        return 1;

    // Set up output vocabulary
    {
        std::string vocabularyFilePath = gDataDirectory + "/" + gOutputVocabularyFileName;
        std::ifstream vocabStream(vocabularyFilePath);
        if (!vocabStream.good())
        {
            printf("Cannot open file %s\n", vocabularyFilePath.c_str());
            return 1;
        }
        vocabStream >> *gOutputVocabulary;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto outputSequenceProperties = getOutputSequenceProperties();
    auto dataReader = getDataReader();
    auto inputEmbedder = getInputEmbedder();
    auto outputEmbedder = getOutputEmbedder();
    auto encoder = getEncoder();
    auto decoder = getDecoder();
    auto alignment = getAlignment();
    auto context = getContext();
    auto attention = getAttention();
    auto projection = getProjection();
    auto likelihood = getLikelihood();
    auto searchPolicy = getSearchPolicy(outputSequenceProperties->getEndSequenceId(), likelihood->getLikelihoodCombinationOperator());
    auto dataWriter = getDataWriter();

    if (gPrintComponentInfo)
    {
        std::cout << "Component Info:" << std::endl;
        std::cout << "- Data Reader: " << dataReader->getInfo() << std::endl;
        std::cout << "- Input Embedder: " << inputEmbedder->getInfo() << std::endl;
        std::cout << "- Output Embedder: " << outputEmbedder->getInfo() << std::endl;
        std::cout << "- Encoder: " << encoder->getInfo() << std::endl;
        std::cout << "- Decoder: " << decoder->getInfo() << std::endl;
        std::cout << "- Alignment: " << alignment->getInfo() << std::endl;
        std::cout << "- Context: " << context->getInfo() << std::endl;
        std::cout << "- Attention: " << attention->getInfo() << std::endl;
        std::cout << "- Projection: " << projection->getInfo() << std::endl;
        std::cout << "- Likelihood: " << likelihood->getInfo() << std::endl;
        std::cout << "- Search Policy: " << searchPolicy->getInfo() << std::endl;
        std::cout << "- Data Writer: " << dataWriter->getInfo() << std::endl;
        std::cout << "End of Component Info" << std::endl;
    }

    // A number of consistency checks between components
    assert(alignment->getSourceStatesSize() == encoder->getMemoryStatesSize());
    assert((!gInitializeDecoderFromEncoderHiddenStates) || (decoder->getStateSizes() == encoder->getStateSizes()));
    assert(projection->getOutputSize() == outputEmbedder->getInputDimensionSize());

    std::vector<int> stateSizes = encoder->getStateSizes();
    
    auto inputHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int>>(gMaxBatchSize * gMaxInputSequenceLength);
    auto inputSequenceLengthsHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int>>(gMaxBatchSize);
    auto maxOutputSequenceLengthsHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int>>(gMaxBatchSize);
    auto outputSequenceLengthsHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int>>(gMaxBatchSize);
    auto outputLikelihoodHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<float>>(gMaxBatchSize * gBeamWidth * gBeamWidth);
    auto outputVocabularyIndicesHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int>>(gMaxBatchSize * gBeamWidth * gBeamWidth);
    auto sourceRayIndicesHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int>>(gMaxBatchSize * gBeamWidth);
    auto sourceRayOptionIndicesHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int>>(gMaxBatchSize * gBeamWidth);

    // Allocated buffers on GPU to be used as inputs and outputs for TenorRT
    auto inputEncoderDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int>>(gMaxBatchSize * gMaxInputSequenceLength);
    auto inputSequenceLengthsDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int>>(gMaxBatchSize);
    auto inputSequenceLengthsReplicatedDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int>>(gMaxBatchSize * gBeamWidth);
    auto memoryStatesDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * gMaxInputSequenceLength * encoder->getMemoryStatesSize());
    auto attentionKeysDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>((alignment->getAttentionKeySize() > 0) ? gMaxBatchSize * gMaxInputSequenceLength * alignment->getAttentionKeySize(): 0);
    std::vector<nmtSample::DeviceBuffer<float>::ptr> encoderStatesLastTimestepDeviceBuffers;
    for(auto stateSize: stateSizes)
        encoderStatesLastTimestepDeviceBuffers.push_back(std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * stateSize));
    std::vector<nmtSample::DeviceBuffer<float>::ptr> inputDecoderStatesDeviceBuffers;
    for(auto stateSize: stateSizes)
        inputDecoderStatesDeviceBuffers.push_back(std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * gBeamWidth * stateSize));
    std::vector<nmtSample::DeviceBuffer<float>::ptr> outputDecoderStatesDeviceBuffers;
    for(auto stateSize: stateSizes)
        outputDecoderStatesDeviceBuffers.push_back(std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * gBeamWidth * stateSize));
    auto inputAttentionDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>(gFeedAttentionToInput ? gMaxBatchSize * gBeamWidth * attention->getAttentionSize() : 0);
    auto outputAttentionDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>(gFeedAttentionToInput ? gMaxBatchSize * gBeamWidth * attention->getAttentionSize() : 0);
    auto outputLikelihoodDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * gBeamWidth * gBeamWidth);
    auto outputVocabularyIndicesDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int>>(gMaxBatchSize * gBeamWidth * gBeamWidth);
    auto sourceRayIndicesDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int>>(gMaxBatchSize * gBeamWidth);
    auto sourceRayOptionIndicesDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int>>(gMaxBatchSize * gBeamWidth);
    auto inputDecoderDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int>>(gMaxBatchSize * gBeamWidth);

    std::vector<nmtSample::DeviceBuffer<float>::ptr> zeroInputEncoderStatesDeviceBuffers;
    for(auto stateSize: stateSizes)
    {
        auto buf = std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * stateSize);
        CUDA_CHECK(cudaMemsetAsync(*buf, 0, gMaxBatchSize * stateSize * sizeof(float), stream));
        zeroInputEncoderStatesDeviceBuffers.push_back(buf);
    }

    std::vector<nmtSample::DeviceBuffer<float>::ptr> zeroInputDecoderStatesDeviceBuffers;
    for(auto stateSize: stateSizes)
    {
        auto buf = std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * gBeamWidth * stateSize);
        CUDA_CHECK(cudaMemsetAsync(*buf, 0, gMaxBatchSize * gBeamWidth * stateSize * sizeof(float), stream));
        zeroInputDecoderStatesDeviceBuffers.push_back(buf);
    }

    auto zeroInputAttentionDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>(gFeedAttentionToInput ? gMaxBatchSize * gBeamWidth * attention->getAttentionSize() : 0);
    if (gFeedAttentionToInput)
    {
        CUDA_CHECK(cudaMemsetAsync(*zeroInputAttentionDeviceBuffer, 0, gMaxBatchSize * gBeamWidth * attention->getAttentionSize() * sizeof(float), stream));
    }
    auto startSeqInputDecoderDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int>>(gMaxBatchSize * gBeamWidth);
    {
        auto startSeqInputDecoderHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int>>(gMaxBatchSize * gBeamWidth);
        std::fill_n((int *)*startSeqInputDecoderHostBuffer, gMaxBatchSize * gBeamWidth, outputSequenceProperties->getStartSequenceId());
        CUDA_CHECK(cudaMemcpyAsync(*startSeqInputDecoderDeviceBuffer, *startSeqInputDecoderHostBuffer, gMaxBatchSize * gBeamWidth * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto zeroInitializeDecoderIndicesDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int>>(gMaxBatchSize * gBeamWidth);
    CUDA_CHECK(cudaMemsetAsync(*zeroInitializeDecoderIndicesDeviceBuffer, 0, gMaxBatchSize * gBeamWidth * sizeof(int), stream));

    // Create TensorRT engines
    nmtSample::TRTEngine::ptr encoderEngine = getEncoderEngine(inputEmbedder, encoder, alignment);
    nmtSample::TRTEngine::ptr generatorEngine = getGeneratorEngine(outputEmbedder, decoder, alignment, context, attention, projection, likelihood);
    nmtSample::TRTEngine::ptr generatorShuffleEngine = getGeneratorShuffleEngine(decoder->getStateSizes(), attention->getAttentionSize());

    // Setup TensorRT bindings
    std::vector<void *> encoderBindings(encoderEngine->getImpl()->getNbBindings());
    encoderBindings[encoderEngine->getImpl()->getBindingIndex("input_encoder_data")] = (int *)(*inputEncoderDeviceBuffer);
    encoderBindings[encoderEngine->getImpl()->getBindingIndex("actual_input_sequence_lengths")] = (int *)(*inputSequenceLengthsDeviceBuffer);
    encoderBindings[encoderEngine->getImpl()->getBindingIndex("actual_input_sequence_lengths_replicated")] = (int *)(*inputSequenceLengthsReplicatedDeviceBuffer);
    encoderBindings[encoderEngine->getImpl()->getBindingIndex("initialize_decoder_indices")] = (int *)(*zeroInitializeDecoderIndicesDeviceBuffer);
    for(int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "input_encoder_states_" << i;
        encoderBindings[encoderEngine->getImpl()->getBindingIndex(ss.str().c_str())] = (float *)(*zeroInputEncoderStatesDeviceBuffers[i]);
    }
    encoderBindings[encoderEngine->getImpl()->getBindingIndex("memory_states")] = (float *)(*memoryStatesDeviceBuffer);
    if (alignment->getAttentionKeySize() > 0)
    {
        encoderBindings[encoderEngine->getImpl()->getBindingIndex("attention_keys")] = (float *)(*attentionKeysDeviceBuffer);
    }
    if (gInitializeDecoderFromEncoderHiddenStates)
    {
        for(int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
        {
            std::stringstream ss;
            ss << "input_decoder_states_" << i;
            encoderBindings[encoderEngine->getImpl()->getBindingIndex(ss.str().c_str())] = (float *)(*inputDecoderStatesDeviceBuffers[i]);
        }
    }

    std::vector<void *> generatorBindings(generatorEngine->getImpl()->getNbBindings());
    generatorBindings[generatorEngine->getImpl()->getBindingIndex("input_decoder_data")] = (int *)(*inputDecoderDeviceBuffer);
    for(int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "input_decoder_states_" << i;
        generatorBindings[generatorEngine->getImpl()->getBindingIndex(ss.str().c_str())] = (float *)(*inputDecoderStatesDeviceBuffers[i]);
    }
    generatorBindings[generatorEngine->getImpl()->getBindingIndex("actual_input_sequence_lengths_replicated")] = (int *)(*inputSequenceLengthsReplicatedDeviceBuffer);
    generatorBindings[generatorEngine->getImpl()->getBindingIndex("memory_states")] = (float *)(*memoryStatesDeviceBuffer);
    if (alignment->getAttentionKeySize() > 0)
    {
        generatorBindings[generatorEngine->getImpl()->getBindingIndex("attention_keys")] = (float *)(*attentionKeysDeviceBuffer);
    }
    for(int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "output_decoder_states_" << i;
        generatorBindings[generatorEngine->getImpl()->getBindingIndex(ss.str().c_str())] = (float *)(*outputDecoderStatesDeviceBuffers[i]);
    }
    generatorBindings[generatorEngine->getImpl()->getBindingIndex("output_likelihoods")] = (float *)(*outputLikelihoodDeviceBuffer);
    generatorBindings[generatorEngine->getImpl()->getBindingIndex("output_vocabulary_indices")] = (int *)(*outputVocabularyIndicesDeviceBuffer);
    if (gFeedAttentionToInput)
    {
        generatorBindings[generatorEngine->getImpl()->getBindingIndex("input_attention")] = (float *)(*inputAttentionDeviceBuffer);
        generatorBindings[generatorEngine->getImpl()->getBindingIndex("output_attention")] = (float *)(*outputAttentionDeviceBuffer);
    }

    std::vector<void *> generatorBindingsFirstStep = generatorBindings;
    generatorBindingsFirstStep[generatorEngine->getImpl()->getBindingIndex("input_decoder_data")] = (int *)(*startSeqInputDecoderDeviceBuffer);
    if (!gInitializeDecoderFromEncoderHiddenStates)
    {
        for(int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
        {
            std::stringstream ss;
            ss << "input_decoder_states_" << i;
            generatorBindingsFirstStep[generatorEngine->getImpl()->getBindingIndex(ss.str().c_str())] = (float *)(*zeroInputDecoderStatesDeviceBuffers[i]);
        }
    }
    if (gFeedAttentionToInput)
    {
        generatorBindingsFirstStep[generatorEngine->getImpl()->getBindingIndex("input_attention")] = (float *)(*zeroInputAttentionDeviceBuffer);
    }

    std::vector<void *> generatorShuffleBindings(generatorShuffleEngine->getImpl()->getNbBindings());
    generatorShuffleBindings[generatorShuffleEngine->getImpl()->getBindingIndex("source_ray_indices")] = (int *)(*sourceRayIndicesDeviceBuffer);
    generatorShuffleBindings[generatorShuffleEngine->getImpl()->getBindingIndex("source_ray_option_indices")] = (int *)(*sourceRayOptionIndicesDeviceBuffer);
    generatorShuffleBindings[generatorShuffleEngine->getImpl()->getBindingIndex("previous_output_vocabulary_indices")] = (int *)(*outputVocabularyIndicesDeviceBuffer);
    for(int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "previous_output_decoder_states_" << i;
        generatorShuffleBindings[generatorShuffleEngine->getImpl()->getBindingIndex(ss.str().c_str())] = (float *)(*outputDecoderStatesDeviceBuffers[i]);
    }
    generatorShuffleBindings[generatorShuffleEngine->getImpl()->getBindingIndex("input_decoder_data")] = (int *)(*inputDecoderDeviceBuffer);
    for(int i = 0; i < static_cast<int>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "input_decoder_states_" << i;
        generatorShuffleBindings[generatorShuffleEngine->getImpl()->getBindingIndex(ss.str().c_str())] = (float *)(*inputDecoderStatesDeviceBuffers[i]);
    }
    if (gFeedAttentionToInput)
    {
        generatorShuffleBindings[generatorShuffleEngine->getImpl()->getBindingIndex("previous_output_attention")] = (float *)(*outputAttentionDeviceBuffer);
        generatorShuffleBindings[generatorShuffleEngine->getImpl()->getBindingIndex("input_attention")] = (float *)(*inputAttentionDeviceBuffer);
    }

    // Craete Tensor RT contexts
    nmtSample::TRTContext encoderContext(encoderEngine->getImpl()->createExecutionContext());
    nmtSample::TRTContext generatorContext(generatorEngine->getImpl()->createExecutionContext());
    nmtSample::TRTContext generatorShuffleContext(generatorShuffleEngine->getImpl()->createExecutionContext());

    dataWriter->initialize();

    std::vector<int> outputHostBuffer;
    int inputSamplesRead = dataReader->read(gMaxBatchSize, gMaxInputSequenceLength, *inputHostBuffer, *inputSequenceLengthsHostBuffer);
    // Outer loop over batches of samples
    while(inputSamplesRead > 0)
    {
        CUDA_CHECK(cudaMemcpyAsync(*inputEncoderDeviceBuffer, *inputHostBuffer, inputSamplesRead * gMaxInputSequenceLength * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(*inputSequenceLengthsDeviceBuffer, *inputSequenceLengthsHostBuffer, inputSamplesRead * sizeof(int), cudaMemcpyHostToDevice, stream));

        encoderContext.getImpl()->enqueue(inputSamplesRead, &encoderBindings[0], stream, nullptr);

        // Limit output sequences length to input_sequence_length * 2 
        std::transform((const int*)*inputSequenceLengthsHostBuffer, (const int*)*inputSequenceLengthsHostBuffer + inputSamplesRead, (int *)*maxOutputSequenceLengthsHostBuffer, [](int i) {int r = i * 2; if (gMaxOutputSequenceLength >= 0) r = std::min(r, gMaxOutputSequenceLength); return r;});
        searchPolicy->initialize(inputSamplesRead, *maxOutputSequenceLengthsHostBuffer);
        int batchMaxOutputSequenceLength = *std::max_element((int *)*maxOutputSequenceLengthsHostBuffer, (int *)*maxOutputSequenceLengthsHostBuffer + inputSamplesRead);
        outputHostBuffer.resize(gMaxBatchSize * batchMaxOutputSequenceLength);

        // Inner loop over generator timesteps
        for(int outputTimestep = 0; (outputTimestep < batchMaxOutputSequenceLength) && searchPolicy->haveMoreWork(); ++outputTimestep)
        {
            // Generator initialization and beam shuffling
            if (outputTimestep == 0)
            {
                generatorContext.getImpl()->enqueue(inputSamplesRead, &generatorBindingsFirstStep[0], stream, nullptr);
            }
            else
            {
                generatorShuffleContext.getImpl()->enqueue(inputSamplesRead, &generatorShuffleBindings[0], stream, nullptr);
                generatorContext.getImpl()->enqueue(inputSamplesRead, &generatorBindings[0], stream, nullptr);
            }

            CUDA_CHECK(cudaMemcpyAsync(*outputLikelihoodHostBuffer, *outputLikelihoodDeviceBuffer, inputSamplesRead * gBeamWidth * gBeamWidth * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(*outputVocabularyIndicesHostBuffer, *outputVocabularyIndicesDeviceBuffer, inputSamplesRead * gBeamWidth * gBeamWidth * sizeof(int), cudaMemcpyDeviceToHost, stream));
            
            CUDA_CHECK(cudaStreamSynchronize(stream));

            searchPolicy->processTimestep(
                inputSamplesRead,
                *outputLikelihoodHostBuffer,
                *outputVocabularyIndicesHostBuffer,
                *sourceRayIndicesHostBuffer,
                *sourceRayOptionIndicesHostBuffer);

            CUDA_CHECK(cudaMemcpyAsync(*sourceRayIndicesDeviceBuffer, *sourceRayIndicesHostBuffer, inputSamplesRead * gBeamWidth * sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(*sourceRayOptionIndicesDeviceBuffer, *sourceRayOptionIndicesHostBuffer, inputSamplesRead * gBeamWidth * sizeof(int), cudaMemcpyHostToDevice, stream));
        } // for(int outputTimestep

        searchPolicy->readGeneratedResult(
            inputSamplesRead,
            batchMaxOutputSequenceLength,
            &outputHostBuffer[0],
            *outputSequenceLengthsHostBuffer);

        dataWriter->write(
            inputSamplesRead,
            batchMaxOutputSequenceLength,
            &outputHostBuffer[0],
            *outputSequenceLengthsHostBuffer);

        inputSamplesRead = dataReader->read(gMaxBatchSize, gMaxInputSequenceLength, *inputHostBuffer, *inputSequenceLengthsHostBuffer);
    }

    dataWriter->finalize();

    cudaStreamDestroy(stream);

    return 0;
}
