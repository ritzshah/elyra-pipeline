# Elyra on OpenShift AI - Demo Walkthrough Guide

## 🎯 Overview
This guide provides a step-by-step walkthrough for demonstrating Elyra on OpenShift AI, covering pipeline development, MLOps automation, KServe model serving, and vLLM optimization.

**Total Demo Time**: 45 minutes
**Audience**: Data Scientists, ML Engineers, DevOps Teams

---

## 🚀 Pre-Demo Setup Checklist

### Before You Start
1. ✅ Run `./setup-demo.sh` and confirm successful completion
2. ✅ Note the access URLs provided by the setup script
3. ✅ Verify all pods are running: `oc get pods -n elyra-demo-project`
4. ✅ Prepare browser tabs with:
   - OpenShift AI Dashboard
   - JupyterLab (Elyra)
   - OpenShift Console (optional)

### Quick Status Check
```bash
# Verify core components
oc get pods -n elyra-demo-project
oc get routes -n elyra-demo-project
oc get pvc -n elyra-demo-project

# Check if GPU nodes available for vLLM demo
oc get nodes --show-labels | grep nvidia.com/gpu
```

---

## 📋 Demo Flow Structure

### Part 1: Elyra Pipeline Development (12 minutes)
### Part 2: MLOps Automation Workflows (12 minutes)  
### Part 3: KServe Model Serving (8 minutes)
### Part 4: vLLM LLM Optimization (10 minutes) *[If GPU available]*
### Part 5: Q&A and Wrap-up (3 minutes)

---

## 🎬 Part 1: Elyra Pipeline Development (12 minutes)

### 1.1 Access JupyterLab with Elyra (2 minutes)

**Script**: *"Let me show you how Elyra extends JupyterLab to create visual ML pipelines."*

1. **Open JupyterLab** using the URL from setup output
2. **Point out Elyra features**:
   - Pipeline editor icon in left sidebar
   - Enhanced notebook interface
   - Git integration panel

**Demo Talking Points**:
- "Elyra seamlessly integrates with JupyterLab"
- "Data scientists work in familiar notebook environment"
- "Visual pipeline editor simplifies workflow orchestration"

### 1.2 Explore Sample Notebooks (3 minutes)

**Script**: *"These notebooks represent typical ML workflow components."*

1. **Navigate to `notebooks/` folder**
2. **Open `01-data-preparation.ipynb`**:
   - Show parameterized cells (tagged with `parameters`)
   - Highlight modular structure
   - Point out pipeline outputs

3. **Briefly show other notebooks**:
   - `02-model-training.ipynb` - Multi-model training
   - `03-model-evaluation.ipynb` - Validation and registry

**Demo Talking Points**:
- "Each notebook is a reusable pipeline component"
- "Parameters make components configurable"
- "Outputs flow between pipeline stages"

### 1.3 Create Visual Pipeline (5 minutes)

**Script**: *"Now let's create a visual pipeline using drag-and-drop."*

1. **Open Pipeline Editor**:
   - Click pipeline icon (🔗) in left sidebar
   - Create new pipeline

2. **Build Pipeline Visually**:
   - Drag `01-data-preparation.ipynb` to canvas
   - Drag `02-model-training.ipynb` to canvas
   - Drag `03-model-evaluation.ipynb` to canvas
   - Connect them with arrows (output → input)

3. **Configure Pipeline Settings**:
   - Right-click each node → Properties
   - Show runtime image configuration
   - Demonstrate parameter passing
   - Set resource requirements

4. **Configure Runtime**:
   - Click ⚙️ (settings) in pipeline editor
   - Show Kubeflow Pipelines runtime connection
   - Point out OpenShift AI integration

**Demo Talking Points**:
- "Visual editor eliminates complex YAML configuration"
- "Drag-and-drop workflow creation"
- "Automatic dependency management"
- "Enterprise-ready with Kubeflow Pipelines"

### 1.4 Pipeline Submission (2 minutes)

**Script**: *"Let's submit this pipeline to OpenShift AI."*

1. **Save Pipeline**: `Ctrl+S` or File → Save
2. **Submit Pipeline**:
   - Click ▶️ (run) button
   - Show submission dialog
   - Configure experiment name: "elyra-demo-run"
   - Click Submit

3. **Monitor Execution**:
   - Switch to Data Science Pipelines UI
   - Show pipeline execution graph
   - Point out real-time status updates

**Demo Talking Points**:
- "One-click submission to production environment"
- "Automatic containerization and orchestration"
- "Real-time monitoring and logging"

---

## ⚙️ Part 2: MLOps Automation Workflows (12 minutes)

### 2.1 Pipeline Execution Monitoring (3 minutes)

**Script**: *"Let's examine how Elyra automates MLOps workflows."*

1. **Access Data Science Pipelines**:
   - Use URL from setup: `https://ds-pipeline-dspa-<namespace>.<domain>`
   - Navigate to Experiments → elyra-demo-run

2. **Show Pipeline Graph**:
   - Real-time execution status
   - Component dependencies
   - Resource utilization
   - Logs and artifacts

**Demo Talking Points**:
- "Kubernetes-native execution"
- "Automatic resource management"
- "Built-in monitoring and observability"

### 2.2 Automated Artifact Management (3 minutes)

**Script**: *"Notice how Elyra automatically manages all ML artifacts."*

1. **Show Artifact Storage**:
   - Click on completed pipeline nodes
   - Show input/output artifacts
   - Browse stored data and models

2. **Model Registry Integration**:
   - Navigate to completed model training step
   - Show generated model artifacts
   - Point out automatic versioning

**Demo Talking Points**:
- "Automatic artifact versioning and storage"
- "Lineage tracking for reproducibility"
- "Integration with model registry"

### 2.3 Model Validation Gates (3 minutes)

**Script**: *"Elyra enforces quality gates before production deployment."*

1. **Open Model Evaluation Results**:
   - Show performance metrics
   - Highlight validation thresholds
   - Point out approval/rejection logic

2. **Show Model Registry**:
   - Browse model versions
   - Show model cards and metadata
   - Demonstrate deployment readiness status

**Demo Talking Points**:
- "Automated quality assurance"
- "Performance threshold enforcement"
- "Governance and compliance built-in"

### 2.4 CI/CD Integration (3 minutes)

**Script**: *"This integrates with your existing CI/CD workflows."*

1. **Show Git Integration**:
   - Demonstrate notebook version control
   - Show pipeline definitions in Git
   - Point out collaborative features

2. **Automated Triggers**:
   - Explain how pipelines can be triggered by:
     - Data changes
     - Model performance degradation
     - Scheduled retraining

**Demo Talking Points**:
- "GitOps workflow integration"
- "Collaborative development"
- "Automated retraining triggers"

---

## 🤖 Part 3: KServe Model Serving (8 minutes)

### 3.1 Model Deployment with KServe (3 minutes)

**Script**: *"Now let's deploy our trained model for production serving."*

1. **Show KServe Configuration**:
   ```bash
   # In terminal or show file
   cat kserve/sklearn-inference-service.yaml
   ```

2. **Deploy Model**:
   ```bash
   oc apply -f kserve/sklearn-inference-service.yaml
   ```

3. **Monitor Deployment**:
   ```bash
   oc get inferenceservice -n elyra-demo-project
   kubectl get pods -l serving.kserve.io/inferenceservice=elyra-demo-model
   ```

**Demo Talking Points**:
- "Kubernetes-native model serving"
- "Automatic scaling and load balancing"
- "Multi-framework support (sklearn, PyTorch, TensorFlow)"

### 3.2 Auto-scaling Configuration (2 minutes)

**Script**: *"KServe provides production-ready auto-scaling."*

1. **Show Auto-scaling Config**:
   ```bash
   cat kserve/autoscaling-config.yaml
   ```

2. **Apply Configuration**:
   ```bash
   oc apply -f kserve/autoscaling-config.yaml
   ```

3. **Demonstrate Scaling**:
   - Show HPA configuration
   - Point out scale-to-zero capability
   - Explain GPU-aware scaling

**Demo Talking Points**:
- "Scale from 0 to N based on demand"
- "Cost-effective resource utilization"
- "GPU-aware auto-scaling for LLMs"

### 3.3 Canary Deployments (3 minutes)

**Script**: *"KServe supports advanced deployment strategies."*

1. **Deploy Canary Version**:
   ```bash
   oc apply -f kserve/canary-deployment.yaml
   ```

2. **Show Traffic Split**:
   - Demonstrate 90/10 traffic split
   - Show A/B testing capabilities
   - Point out automated rollback triggers

3. **Test Inference**:
   ```bash
   # Show sample inference request
   curl -X POST https://elyra-demo-model.<domain>/v1/models/elyra-demo-model:predict \
     -H "Content-Type: application/json" \
     -d '{"instances": [[1.0, 2.0, 3.0, 0]]}'
   ```

**Demo Talking Points**:
- "Safe production deployments"
- "Automated rollback on errors"
- "Real-time monitoring and alerting"

---

## 🔥 Part 4: vLLM LLM Optimization (10 minutes) *[GPU Required]*

### 4.1 vLLM Architecture Overview (2 minutes)

**Script**: *"For LLM workloads, we use vLLM for optimized inference."*

1. **Show vLLM Configuration**:
   ```bash
   cat vllm/vllm-deployment.yaml
   ```

2. **Highlight Optimizations**:
   - GPU memory utilization (85% for T4)
   - Batching strategies
   - Prefix caching
   - FP16 precision

**Demo Talking Points**:
- "Optimized for GPU inference"
- "Memory-efficient attention mechanisms"
- "Industry-leading throughput for LLMs"

### 4.2 Deploy vLLM Service (3 minutes)

**Script**: *"Let's deploy a 7B parameter model on our GPU (T4/L4)."*

1. **Deploy vLLM**:
   ```bash
   oc apply -f vllm/vllm-deployment.yaml
   ```

2. **Monitor GPU Utilization**:
   ```bash
   # Show GPU monitoring
   oc get pods -l app=vllm-llama-model
   oc logs -f deployment/vllm-llama-model
   ```

3. **Wait for Model Loading**:
   - Show loading progress in logs
   - Explain model quantization
   - Point out memory optimization

**Demo Talking Points**:
- "Efficient model loading and caching"
- "Automatic GPU memory management"
- "Production-ready LLM serving"

### 4.3 Performance Optimization (3 minutes)

**Script**: *"vLLM provides extensive performance tuning options."*

1. **Show Optimization Config**:
   ```bash
   cat vllm/vllm-optimization-config.yaml
   ```

2. **Key Optimization Areas**:
   - Batching configuration
   - Memory management
   - Scheduling policies
   - CUDA graph optimization

3. **Monitoring Dashboard**:
   - Show Grafana dashboard (if available)
   - GPU utilization metrics
   - Throughput and latency graphs

**Demo Talking Points**:
- "Configurable for different workload patterns"
- "Continuous batching for efficiency"
- "Real-time performance monitoring"

### 4.4 Benchmarking (2 minutes)

**Script**: *"Let's test the inference performance."*

1. **Run Benchmark Script**:
   ```bash
   python vllm/benchmarking_script.py
   ```

2. **Show Results**:
   - Requests per second
   - Tokens per second
   - Latency percentiles
   - GPU utilization

3. **Compare with Baseline**:
   - Show performance improvements
   - Cost optimization benefits
   - Scalability advantages

**Demo Talking Points**:
- "Measurable performance improvements"
- "Cost-effective LLM deployment"
- "Production-ready performance"

---

## 🎯 Part 5: Q&A and Wrap-up (3 minutes)

### 5.1 Key Takeaways Summary

**Script**: *"Let's summarize what we've demonstrated today."*

**For Data Scientists**:
- ✅ Visual pipeline development with familiar notebooks
- ✅ Automated MLOps without infrastructure complexity
- ✅ One-click deployment to production

**For ML Engineers**:
- ✅ Kubernetes-native scaling and reliability
- ✅ Advanced deployment strategies (canary, A/B testing)
- ✅ Optimized LLM inference with vLLM

**For DevOps Teams**:
- ✅ GitOps integration and CI/CD workflows
- ✅ Cost-effective resource management
- ✅ Enterprise-grade monitoring and governance

### 5.2 Next Steps

1. **Try the Demo Environment**:
   - Access details provided in setup output
   - Experiment with your own notebooks
   - Modify pipeline configurations

2. **Production Deployment**:
   - Scale to multiple GPU nodes
   - Integrate with your data sources
   - Configure production monitoring

3. **Community and Support**:
   - Elyra GitHub: https://github.com/elyra-ai/elyra
   - OpenShift AI Documentation
   - Red Hat Support (for OpenShift AI customers)

---

## 🛠️ Troubleshooting During Demo

### Common Issues and Quick Fixes

**Pipeline Fails to Submit**:
```bash
# Check runtime configuration
oc get pods -n elyra-demo-project | grep pipeline
oc logs deployment/ds-pipeline-dspa -n elyra-demo-project
```

**Model Serving Not Starting**:
```bash
# Check KServe status
oc get inferenceservice -n elyra-demo-project
oc describe inferenceservice elyra-demo-model -n elyra-demo-project
```

**vLLM Out of Memory**:
```bash
# Check GPU memory
oc exec -it <vllm-pod> -- nvidia-smi
# Reduce gpu-memory-utilization in deployment
```

**Notebooks Not Loading**:
```bash
# Check JupyterLab pod
oc get pods -l app=elyra-demo-notebook
oc logs deployment/elyra-demo-notebook -n elyra-demo-project
```

### Backup Demo Strategy

If live demo fails, have these ready:
- Screenshots of working pipeline
- Pre-recorded video clips
- Sample outputs and metrics
- Architecture diagrams

---

## 📊 Demo Success Metrics

### Audience Engagement Indicators
- Questions about specific features
- Requests for deeper technical details
- Interest in pilot projects
- Follow-up meeting requests

### Technical Demonstration Goals
- ✅ Show end-to-end ML workflow
- ✅ Demonstrate production readiness
- ✅ Highlight OpenShift AI integration
- ✅ Prove scalability and performance

---

## 🎬 Presenter Tips

### Timing Management
- **Set expectations**: "45-minute comprehensive demo"
- **Use timers**: Keep each section on track
- **Buffer time**: Leave 5 minutes for overruns

### Technical Preparation
- **Test everything**: Run full demo beforehand
- **Have backups**: Screenshots, videos, alternative demos
- **Know the architecture**: Be ready for deep technical questions

### Audience Engagement
- **Ask questions**: "How many are using Kubeflow today?"
- **Share experiences**: "In production, we've seen..."
- **Be interactive**: "What would you like to see next?"

### Handling Issues
- **Stay calm**: Technical issues happen
- **Have alternatives**: Move to next section if something fails
- **Use it as teaching**: "This shows why monitoring is important"

---

**Good luck with your demo! 🚀**

This guide should help you deliver a compelling demonstration of Elyra's capabilities on OpenShift AI.