# 🚀 Deployment Guide - Bootcamp Microsoft Data Scientist Azure

Este guia demonstra como fazer deploy do modelo de predição de risco de crédito no Azure ML para o Bootcamp Microsoft.

## 📋 Pré-requisitos

- Azure CLI instalado e configurado
- Extensão Azure ML: `az extension add -n ml`
- Workspace do Azure ML criado
- Modelo treinado localmente

## 🔧 Setup Inicial

### 1. Login e Configuração
```bash
# Login no Azure
az login

# Definir subscription
az account set --subscription "YOUR_SUBSCRIPTION_ID"

# Criar resource group (se não existir)
az group create --name rg-bootcamp-azure-demo --location eastus
```

### 2. Criar Workspace
```bash
# Criar workspace Azure ML
az ml workspace create \
    --name bootcamp-azure-workspace \
    --resource-group rg-bootcamp-azure-demo \
    --location eastus
```

### 3. Criar Compute Cluster
```bash
# Criar compute para treinamento
az ml compute create \
    --name cpu-cluster \
    --type amlcompute \
    --min-instances 0 \
    --max-instances 4 \
    --size Standard_DS3_v2
```

## 🏋️ Treinamento no Azure ML

### Opção 1: Job Individual
```bash
# Submeter job de treinamento
az ml job create --file azure-ml/jobs/train-job.yml
```

### Opção 2: Pipeline Completo
```bash
# Submeter pipeline de treinamento
az ml job create --file azure-ml/pipelines/training-pipeline.yml
```

### Monitorar Jobs
```bash
# Listar jobs
az ml job list

# Ver detalhes de um job
az ml job show --name JOB_NAME
```

## 🌐 Deploy do Modelo

### 1. Criar Endpoint Online
```bash
# Criar endpoint
az ml online-endpoint create --file azure-ml/endpoints/credit-endpoint.yml

# Verificar status
az ml online-endpoint show --name credit-risk-endpoint
```

### 2. Deploy do Modelo
```bash
# Deploy modelo no endpoint
az ml online-deployment create --file azure-ml/deployments/xgboost-deployment.yml

# Verificar deployment
az ml online-deployment show \
    --name blue \
    --endpoint-name credit-risk-endpoint
```

### 3. Testar Endpoint
```bash
# Testar com dados de exemplo
az ml online-endpoint invoke \
    --name credit-risk-endpoint \
    --request-file test-data.json
```

## 📊 Monitoramento

### Application Insights
```bash
# Habilitar Application Insights
az ml online-endpoint update \
    --name credit-risk-endpoint \
    --application-insights-enabled true
```

### Logs do Deployment
```bash
# Ver logs
az ml online-deployment get-logs \
    --name blue \
    --endpoint-name credit-risk-endpoint
```

## 🔄 MLOps Pipeline

### Continuous Integration
```bash
# O GitHub Actions pipeline inclui:
# 1. Testes automatizados
# 2. Validação de modelos  
# 3. Deploy automático para staging
# 4. Aprovação manual para produção
```

### Model Registry
```bash
# Registrar modelo
az ml model create \
    --name credit-risk-model \
    --version 1 \
    --path outputs/xgboost_model.pkl \
    --type mlflow_model
```

## 🔐 Segurança

### Managed Identity
```bash
# Configurar managed identity para o endpoint
az ml online-endpoint update \
    --name credit-risk-endpoint \
    --auth-mode aad_token
```

### Network Security
```bash
# Deploy em VNet (opcional)
az ml online-endpoint create \
    --file azure-ml/endpoints/credit-endpoint-vnet.yml
```

## 📈 Scaling

### Auto-scaling
```yaml
# No deployment YAML:
scale_settings:
  type: target_utilization
  target_utilization_percentage: 70
  min_instances: 1
  max_instances: 10
```

### Multi-region Deployment
```bash
# Deploy em múltiplas regiões
az ml online-endpoint create \
    --name credit-risk-global \
    --traffic "eastus=50,westus=50"
```

## 🧪 A/B Testing

### Blue-Green Deployment
```bash
# Deploy nova versão (green)
az ml online-deployment create --file azure-ml/deployments/xgboost-v2-deployment.yml

# Dividir tráfego
az ml online-endpoint update \
    --name credit-risk-endpoint \
    --traffic "blue=90,green=10"

# Após validação, migrar totalmente
az ml online-endpoint update \
    --name credit-risk-endpoint \
    --traffic "blue=0,green=100"
```

## 🚨 Troubleshooting

### Problemas Comuns

1. **Endpoint não responde**
   ```bash
   # Verificar logs
   az ml online-deployment get-logs --name blue --endpoint-name credit-risk-endpoint --lines 100
   ```

2. **Erro de autenticação**
   ```bash
   # Verificar permissões
   az ml online-endpoint show-keys --name credit-risk-endpoint
   ```

3. **Performance baixa**
   ```bash
   # Aumentar instâncias
   az ml online-deployment update --name blue --instance-count 3
   ```

### Health Checks
```bash
# Verificar saúde do endpoint
curl -X GET "https://credit-risk-endpoint.{region}.inference.ml.azure.com/health" \
     -H "Authorization: Bearer $TOKEN"
```

## 💰 Otimização de Custos

### Compute Otimizado
```bash
# Usar instâncias menores para desenvolvimento
az ml compute create \
    --name dev-cluster \
    --type amlcompute \
    --size Standard_DS2_v2 \
    --min-instances 0 \
    --max-instances 1
```

### Scheduled Scaling
```python
# Função para escalar baseado em horários
def scale_endpoint_by_schedule():
    # Durante horário comercial: min=2, max=10
    # Durante madrugada: min=0, max=2
    pass
```

## 📚 Recursos Adicionais

- [Azure ML Documentation](https://docs.microsoft.com/azure/machine-learning/)
- [MLOps Best Practices](https://docs.microsoft.com/azure/machine-learning/concept-model-management-and-deployment)
- [Cost Optimization](https://docs.microsoft.com/azure/machine-learning/how-to-manage-optimize-cost)

---

**Próximos passos após deployment:**

1. 📊 Configurar dashboards de monitoramento
2. 🚨 Implementar alertas de performance
3. 🔄 Setup de re-treino automático
4. 📈 Otimização contínua de hiperparâmetros
5. 🧪 Implementação de testes A/B
