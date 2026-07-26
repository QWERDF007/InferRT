#include <inferrt/core/Exception.hpp>
#include <inferrt/engine/Pipeline.hpp>

#include <algorithm>
#include <utility>

namespace irt::engine {
namespace {

size_t elementSize(const TensorDataType data_type)
{
    switch (data_type)
    {
    case TensorDataType::U8:
        return sizeof(uint8_t);
    case TensorDataType::F32:
        return sizeof(float);
    case TensorDataType::I32:
        return sizeof(int32_t);
    case TensorDataType::I64:
        return sizeof(int64_t);
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported tensor data type");
}

int stageOrder(const PipelineStage stage)
{
    return static_cast<int>(stage);
}

bool isModelTensor(const std::string &name)
{
    return name.starts_with("model.");
}

bool isResultTensor(const std::string &name)
{
    return name.starts_with("result.");
}

void validateStageMemory(const PipelinePlan::Node &node, const std::unordered_map<std::string, TensorDesc> &tensors,
                         const std::unordered_map<std::string, bool> &result_names)
{
    const auto validateDeclared = [&](const std::string &name, const bool input)
    {
        if (isModelTensor(name))
        {
            if (input && node.contract.stage != PipelineStage::CUDA_POSTPROCESS
                && node.contract.stage != PipelineStage::CPU_POSTPROCESS)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "TensorRT output %s is only available to postprocess nodes", name.c_str());
            }
            return;
        }
        if (isResultTensor(name))
        {
            const std::string result_name = name.substr(std::string("result.").size());
            if (!input || node.contract.stage != PipelineStage::CPU_POSTPROCESS || !result_names.contains(result_name))
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Pipeline result tensor %s is only available to CPU postprocess nodes",
                                     name.c_str());
            }
            return;
        }

        const auto found = tensors.find(name);
        if (found == tensors.end())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Pipeline tensor is not declared: %s",
                                 name.c_str());
        }
        const auto memory = found->second.memory_kind;
        switch (node.contract.stage)
        {
        case PipelineStage::CPU_PREPROCESS:
            if (memory != MemoryKind::HOST)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "CPU preprocess tensor must be host memory: %s", name.c_str());
            }
            break;
        case PipelineStage::H2D:
            if ((input && memory != MemoryKind::HOST) || (!input && memory != MemoryKind::DEVICE))
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "H2D operator must transfer host inputs to device outputs: %s", name.c_str());
            }
            break;
        case PipelineStage::CUDA_PREPROCESS:
        case PipelineStage::CUDA_POSTPROCESS:
            if (memory != MemoryKind::DEVICE)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "CUDA pipeline tensor must be device memory: %s", name.c_str());
            }
            break;
        case PipelineStage::CPU_POSTPROCESS:
            if (memory != MemoryKind::HOST)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "CPU postprocess tensor must be host memory: %s", name.c_str());
            }
            break;
        }
    };

    for (const auto &name : node.config.inputs)
    {
        validateDeclared(name, true);
    }
    for (const auto &name : node.config.outputs)
    {
        validateDeclared(name, false);
    }
}

} // namespace

size_t TensorDesc::bytesPerRequest() const
{
    validate("tensor");
    return static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(channels)
         * elementSize(data_type);
}

void TensorDesc::validate(const std::string &name) const
{
    if (width <= 0 || height <= 0 || channels <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Tensor %s must have positive width, height, and channels", name.c_str());
    }
    if (layout == TensorLayout::NCHW && channels <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Tensor %s has invalid NCHW channels", name.c_str());
    }
}

void *TensorView::dataForRequest(const int request_index)
{
    if (request_index < 0 || data == nullptr)
    {
        return nullptr;
    }
    return static_cast<uint8_t *>(data) + static_cast<size_t>(request_index) * bytes_per_request;
}

const void *TensorView::dataForRequest(const int request_index) const
{
    if (request_index < 0 || data == nullptr)
    {
        return nullptr;
    }
    return static_cast<const uint8_t *>(data) + static_cast<size_t>(request_index) * bytes_per_request;
}

bool OperatorRegistry::registerCreator(std::string type, OperatorCreator creator)
{
    if (frozen_ || type.empty() || !creator || creators_.contains(type))
    {
        return false;
    }
    creators_.emplace(std::move(type), std::move(creator));
    return true;
}

void OperatorRegistry::freeze()
{
    frozen_ = true;
}

std::unique_ptr<IOperator> OperatorRegistry::create(const std::string_view type, const NodeConfig &config) const
{
    const auto found = creators_.find(std::string(type));
    if (found == creators_.end())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unknown pipeline operator: %.*s",
                             static_cast<int>(type.size()), type.data());
    }
    auto op = found->second(config);
    if (!op)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "Pipeline operator creator returned null: %.*s",
                             static_cast<int>(type.size()), type.data());
    }
    return op;
}

std::vector<std::string> OperatorRegistry::types() const
{
    std::vector<std::string> result;
    result.reserve(creators_.size());
    for (const auto &[type, _] : creators_)
    {
        result.push_back(type);
    }
    std::sort(result.begin(), result.end());
    return result;
}

bool OperatorRegistry::frozen() const noexcept
{
    return frozen_;
}

PipelineBuilder &PipelineBuilder::addTensor(std::string name, TensorDesc desc)
{
    if (name.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Pipeline tensor name must not be empty");
    }
    desc.validate(name);
    if (!tensors_.emplace(std::move(name), desc).second)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Pipeline tensor is already declared");
    }
    return *this;
}

PipelineBuilder &PipelineBuilder::addNode(std::string type, NodeConfig config)
{
    if (type.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Pipeline operator type must not be empty");
    }
    nodes_.push_back({std::move(type), std::move(config)});
    return *this;
}

PipelineBuilder &PipelineBuilder::addResult(std::string tensor_name, std::string result_name)
{
    if (tensor_name.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Pipeline result tensor name must not be empty");
    }
    if (result_name.empty())
    {
        result_name = tensor_name;
    }
    if (result_name.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Pipeline result name must not be empty");
    }
    results_.emplace_back(std::move(tensor_name), std::move(result_name));
    return *this;
}

PipelineBuilder &PipelineBuilder::setModelInput(std::string tensor_name)
{
    model_input_ = tensor_name;
    model_inputs_.clear();
    model_inputs_.emplace_back(std::move(tensor_name), std::string{});
    return *this;
}

PipelineBuilder &PipelineBuilder::bindModelInput(std::string tensor_name, std::string engine_tensor_name)
{
    if (tensor_name.empty() || engine_tensor_name.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Pipeline and TensorRT input names must not be empty");
    }
    if (model_input_.empty())
    {
        model_input_ = tensor_name;
    }
    model_inputs_.emplace_back(std::move(tensor_name), std::move(engine_tensor_name));
    return *this;
}

std::shared_ptr<const PipelinePlan> PipelineBuilder::build(std::shared_ptr<OperatorRegistry> registry) const
{
    if (!registry)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Pipeline requires an OperatorRegistry");
    }
    std::vector<PipelinePlan::ModelInputBinding> model_inputs;
    if (model_inputs_.empty())
    {
        model_inputs.push_back({model_input_, {}});
    }
    else
    {
        model_inputs.reserve(model_inputs_.size());
        for (const auto &[tensor_name, engine_tensor_name] : model_inputs_)
        {
            model_inputs.push_back({tensor_name, engine_tensor_name});
        }
    }
    if (model_input_.empty() || model_inputs.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Pipeline model input tensor is not declared");
    }
    std::unordered_map<std::string, bool> bound_pipeline_tensors;
    std::unordered_map<std::string, bool> bound_engine_tensors;
    for (const auto &binding : model_inputs)
    {
        const auto model_input = tensors_.find(binding.tensor_name);
        if (model_input == tensors_.end())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Pipeline model input tensor is not declared: %s",
                                 binding.tensor_name.c_str());
        }
        if (model_input->second.memory_kind != MemoryKind::DEVICE
            || model_input->second.data_type != TensorDataType::F32 || model_input->second.layout != TensorLayout::NCHW)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Pipeline model input must be a device float32 NCHW tensor: %s",
                                 binding.tensor_name.c_str());
        }
        if (!bound_pipeline_tensors.emplace(binding.tensor_name, true).second)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Pipeline model input is bound more than once: %s", binding.tensor_name.c_str());
        }
        if (!binding.engine_tensor_name.empty()
            && !bound_engine_tensors.emplace(binding.engine_tensor_name, true).second)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "TensorRT model input is bound more than once: %s",
                                 binding.engine_tensor_name.c_str());
        }
    }

    std::unordered_map<std::string, size_t> producer;
    for (size_t index = 0; index < nodes_.size(); ++index)
    {
        const auto &config = nodes_[index].config;
        for (const auto &input : config.inputs)
        {
            if (!isModelTensor(input) && !isResultTensor(input) && !tensors_.contains(input))
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Pipeline input tensor is not declared: %s",
                                     input.c_str());
            }
        }
        for (const auto &output : config.outputs)
        {
            if (isModelTensor(output) || isResultTensor(output) || !tensors_.contains(output))
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Pipeline output tensor is not declared: %s",
                                     output.c_str());
            }
            if (!producer.emplace(output, index).second)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Pipeline tensor has multiple producers: %s",
                                     output.c_str());
            }
        }
    }

    std::vector<PipelinePlan::ResultBinding> results;
    results.reserve(results_.size());
    std::unordered_map<std::string, bool> result_names;
    for (const auto &[tensor_name, result_name] : results_)
    {
        const auto tensor = tensors_.find(tensor_name);
        if (tensor == tensors_.end())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Pipeline result tensor is not declared: %s",
                                 tensor_name.c_str());
        }
        if (tensor->second.memory_kind != MemoryKind::DEVICE)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Pipeline result tensor must be a device tensor: %s", tensor_name.c_str());
        }
        if (!producer.contains(tensor_name))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Pipeline result tensor must be produced by a pipeline node: %s", tensor_name.c_str());
        }
        if (!result_names.emplace(result_name, true).second)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Pipeline result name is duplicated: %s",
                                 result_name.c_str());
        }
        results.push_back({tensor_name, result_name});
    }

    std::vector<std::vector<size_t>> edges(nodes_.size());
    std::vector<size_t>              indegree(nodes_.size(), 0);
    for (size_t index = 0; index < nodes_.size(); ++index)
    {
        for (const auto &input : nodes_[index].config.inputs)
        {
            const auto found = producer.find(input);
            if (found != producer.end())
            {
                edges[found->second].push_back(index);
                ++indegree[index];
            }
        }
    }

    std::vector<OperatorContract> contracts;
    contracts.reserve(nodes_.size());
    for (const auto &node : nodes_)
    {
        auto op       = registry->create(node.type, node.config);
        auto contract = op->contract();
        if (contract.input_count != node.config.inputs.size() || contract.output_count != node.config.outputs.size())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Pipeline operator %s has incompatible input/output count", node.type.c_str());
        }
        if (contract.kind == ExecutionKind::CPU && contract.stage != PipelineStage::CPU_PREPROCESS
            && contract.stage != PipelineStage::CPU_POSTPROCESS)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "CPU pipeline operator has an invalid execution stage");
        }
        if (contract.kind == ExecutionKind::CUDA && contract.stage != PipelineStage::H2D
            && contract.stage != PipelineStage::CUDA_PREPROCESS && contract.stage != PipelineStage::CUDA_POSTPROCESS)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "CUDA pipeline operator has an invalid execution stage");
        }
        contracts.push_back(contract);
    }

    std::vector<size_t> ready;
    for (size_t index = 0; index < indegree.size(); ++index)
    {
        if (indegree[index] == 0)
        {
            ready.push_back(index);
        }
    }

    std::vector<PipelinePlan::Node> sorted;
    sorted.reserve(nodes_.size());
    int previous_stage = -1;
    while (!ready.empty())
    {
        const auto   found = std::min_element(ready.begin(), ready.end(),
                                              [&contracts](const size_t lhs, const size_t rhs)
                                              {
                                                const int lhs_stage = stageOrder(contracts[lhs].stage);
                                                const int rhs_stage = stageOrder(contracts[rhs].stage);
                                                return lhs_stage == rhs_stage ? lhs < rhs : lhs_stage < rhs_stage;
                                            });
        const size_t index = *found;
        ready.erase(found);

        const auto &contract = contracts[index];
        if (stageOrder(contract.stage) < previous_stage)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Pipeline DAG crosses execution stages backwards; declare transfers explicitly");
        }
        previous_stage = stageOrder(contract.stage);
        sorted.push_back({nodes_[index].type, nodes_[index].config, contract});

        for (const size_t next : edges[index])
        {
            if (--indegree[next] == 0)
            {
                ready.push_back(next);
            }
        }
    }
    if (sorted.size() != nodes_.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Pipeline graph contains a cycle");
    }

    std::unordered_map<std::string, bool> result_name_map;
    for (const auto &binding : results)
    {
        result_name_map.emplace(binding.result_name, true);
    }
    for (const auto &node : sorted)
    {
        validateStageMemory(node, tensors_, result_name_map);
    }

    registry->freeze();
    return std::shared_ptr<const PipelinePlan>(new PipelinePlan(
        std::move(registry), tensors_, std::move(sorted), std::move(results), model_input_, std::move(model_inputs)));
}

PipelinePlan::PipelinePlan(std::shared_ptr<const OperatorRegistry>     registry,
                           std::unordered_map<std::string, TensorDesc> tensors, std::vector<Node> nodes,
                           std::vector<ResultBinding> results, std::string model_input,
                           std::vector<ModelInputBinding> model_inputs)
    : registry_(std::move(registry))
    , tensors_(std::move(tensors))
    , nodes_(std::move(nodes))
    , results_(std::move(results))
    , model_input_(std::move(model_input))
    , model_inputs_(std::move(model_inputs))
{
}

const std::unordered_map<std::string, TensorDesc> &PipelinePlan::tensors() const noexcept
{
    return tensors_;
}

const std::vector<PipelinePlan::Node> &PipelinePlan::nodes() const noexcept
{
    return nodes_;
}

const std::vector<PipelinePlan::ResultBinding> &PipelinePlan::results() const noexcept
{
    return results_;
}

const std::string &PipelinePlan::modelInput() const noexcept
{
    return model_input_;
}

const std::vector<PipelinePlan::ModelInputBinding> &PipelinePlan::modelInputs() const noexcept
{
    return model_inputs_;
}

std::vector<std::unique_ptr<IOperator>> PipelinePlan::createOperators() const
{
    std::vector<std::unique_ptr<IOperator>> operators;
    operators.reserve(nodes_.size());
    for (const auto &node : nodes_)
    {
        auto op = registry_->create(node.type, node.config);
        op->prepare();
        operators.push_back(std::move(op));
    }
    return operators;
}

} // namespace irt::engine
