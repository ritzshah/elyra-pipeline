#!/bin/bash

# Elyra on OpenShift AI Demo Setup Script
# This script sets up the complete demo environment

set -e

# Configuration
NAMESPACE="elyra-demo-project"
# Auto-detect current user
if OC_USER=$(oc whoami 2>/dev/null); then
    echo "🔍 Detected OpenShift user: $OC_USER"
else
    echo "❌ Could not detect current user"
    exit 1
fi

echo "🚀 Setting up Elyra on OpenShift AI Demo"
echo "========================================="

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v oc &> /dev/null; then
    echo "❌ OpenShift CLI (oc) is required but not installed"
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is required but not installed"  
    exit 1
fi

# Check if logged into OpenShift
if ! oc whoami &> /dev/null; then
    echo "❌ Please log in to OpenShift cluster first: oc login"
    exit 1
fi

# Auto-detect cluster domain from OpenShift console route
echo "🔍 Auto-detecting cluster domain..."
if CLUSTER_DOMAIN=$(oc get route console -n openshift-console -o jsonpath='{.spec.host}' 2>/dev/null | sed 's/^console-openshift-console\.//'); then
    echo "✅ Detected cluster domain: $CLUSTER_DOMAIN"
else
    echo "⚠️  Could not auto-detect cluster domain, trying alternative method..."
    if CLUSTER_DOMAIN=$(oc get ingresses.config.openshift.io cluster -o jsonpath='{.spec.domain}' 2>/dev/null); then
        echo "✅ Detected cluster domain: $CLUSTER_DOMAIN"
    else
        echo "❌ Could not auto-detect cluster domain. Please set CLUSTER_DOMAIN environment variable"
        echo "   Example: export CLUSTER_DOMAIN=apps.mycluster.example.com"
        exit 1
    fi
fi

echo "✅ Prerequisites check passed"

# Function to wait for resource
wait_for_resource() {
    local resource_type=$1
    local resource_name=$2
    local namespace=$3
    local timeout=${4:-300}
    
    echo "⏳ Waiting for $resource_type/$resource_name in namespace $namespace..."
    oc wait --for=condition=available $resource_type/$resource_name -n $namespace --timeout=${timeout}s
}

# Function to apply yaml with error handling
apply_yaml() {
    local file=$1
    echo "📝 Applying $file..."
    if oc apply -f "$file"; then
        echo "✅ Successfully applied $file"
    else
        echo "❌ Failed to apply $file"
        exit 1
    fi
}

# Step 1: Create namespace and basic setup
echo "📁 Creating namespace and basic setup..."
apply_yaml "openshift-configs/data-science-project.yaml"

# Wait for namespace to be ready
sleep 10

# Step 2: Set up Data Science Pipelines
echo "🔧 Setting up Data Science Pipelines..."
echo "⏳ Waiting for Data Science Pipelines to be ready (this may take several minutes)..."
sleep 30
# Wait for the DSPA to be ready instead of specific deployment
echo "⏳ Checking DataSciencePipelinesApplication status..."
timeout=600
elapsed=0
while [ $elapsed -lt $timeout ]; do
    if oc get datasciencepipelinesapplication elyra-demo-dspa -n $NAMESPACE -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null | grep -q "True"; then
        echo "✅ Data Science Pipelines are ready!"
        break
    fi
    echo "   Still waiting for Data Science Pipelines... ($elapsed/${timeout}s)"
    sleep 30
    elapsed=$((elapsed + 30))
done

if [ $elapsed -ge $timeout ]; then
    echo "⚠️  Data Science Pipelines deployment is taking longer than expected"
    echo "   Continuing with setup, but pipelines may not be immediately available"
fi

# Step 3: Set up JupyterHub with Elyra
echo "📓 Setting up JupyterHub with Elyra..."
apply_yaml "openshift-configs/jupyter-elyra-setup.yaml"

# Step 4: Set up Model Serving
echo "🤖 Setting up Model Serving..."
apply_yaml "openshift-configs/model-serving-setup.yaml"

# Step 5: Set up KServe inference services
echo "⚡ Setting up KServe inference services..."
apply_yaml "kserve/sklearn-inference-service.yaml"
apply_yaml "kserve/autoscaling-config.yaml"

# Step 6: Set up vLLM (optional, requires GPU nodes)
if oc get nodes --show-labels | grep -q nvidia.com/gpu || oc get nodes -l accelerator --no-headers 2>/dev/null | wc -l | grep -q -v '^0$'; then
    echo "🔥 GPU nodes detected, setting up vLLM..."
    apply_yaml "vllm/vllm-deployment.yaml"
    apply_yaml "vllm/vllm-optimization-config.yaml"
    apply_yaml "vllm/vllm-monitoring.yaml"
else
    echo "⚠️  No GPU nodes detected, skipping vLLM setup"
    echo "   To set up vLLM later, ensure GPU nodes are available and run:"
    echo "   oc apply -f vllm/"
fi

# Step 7: Create demo user and permissions
echo "👤 Setting up demo user permissions..."
oc adm policy add-role-to-user edit $OC_USER -n $NAMESPACE

# Step 8: Upload sample notebooks (if running from git repo)
if [ -d "notebooks" ]; then
    echo "📚 Sample notebooks are available in the notebooks/ directory"
    echo "   Copy these to your JupyterLab environment to get started"
fi

# Step 9: Display access information
echo ""
echo "🎉 Demo setup completed successfully!"
echo "====================================="
echo ""
echo "Access Information:"
echo "📊 OpenShift AI Dashboard: https://rhods-dashboard-redhat-ods-applications.$CLUSTER_DOMAIN"
echo "📓 JupyterLab (Elyra): https://elyra-demo-notebook-$NAMESPACE.$CLUSTER_DOMAIN"
echo "🔧 Data Science Pipelines: https://ds-pipeline-dspa-$NAMESPACE.$CLUSTER_DOMAIN"
echo "📈 Model Serving: Available through OpenShift AI Dashboard"
echo ""
echo "Getting Started:"
echo "1. Access JupyterLab using the URL above"
echo "2. Open the notebooks/ folder"
echo "3. Start with 01-data-preparation.ipynb"
echo "4. Create visual pipelines using the Elyra pipeline editor"
echo "5. Deploy models using KServe"
echo ""
echo "Demo Components:"
echo "✅ Data Science Project: $NAMESPACE"
echo "✅ Kubeflow Pipelines: Configured and ready"
echo "✅ Elyra Runtime: Connected to Data Science Pipelines"
echo "✅ Model Serving: KServe with auto-scaling"
echo "✅ Monitoring: Prometheus metrics configured"

# Check GPU setup
if oc get nodes --show-labels | grep -q nvidia.com/gpu || oc get nodes -l accelerator --no-headers 2>/dev/null | wc -l | grep -q -v '^0$'; then
    echo "✅ vLLM: Ready for LLM inference (GPU enabled)"
else
    echo "⚠️  vLLM: Not configured (requires GPU nodes)"
fi

echo ""
echo "Troubleshooting:"
echo "- Check pod status: oc get pods -n $NAMESPACE"
echo "- View logs: oc logs -f deployment/<deployment-name> -n $NAMESPACE"
echo "- Access pipeline UI: Check Data Science Pipelines in OpenShift AI Dashboard"
echo ""
echo "For more information, see the README.md file"