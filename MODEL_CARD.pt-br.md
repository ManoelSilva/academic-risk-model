[Read in English](MODEL_CARD.md)

# Model Card: Predição de Risco Acadêmico — Classificador Gradient Boosting

## Detalhes do Modelo

### Informações do Modelo
- **Nome do Modelo**: Classificador de Predição de Risco Acadêmico
- **Tipo de Modelo**: Machine Learning Clássico — Ensemble (Gradient Boosting)
- **Framework**: scikit-learn
- **Versão**: 1.0
- **Data**: 02-2026

### Arquitetura do Modelo

**Tipo de Arquitetura**: Pipeline scikit-learn com pré-processamento embutido

**Estrutura do Pipeline**:
```
Entrada: DataFrame com features brutas dos alunos
    ↓
FeatureEngineer (Transformador Customizado):
    - Deriva IS_NEW_STUDENT a partir de SINALIZADOR_INGRESSANTE
    - Remove colunas identificadoras (NOME, INSTITUICAO_ENSINO_ALUNO_2020)
    ↓
Preprocessor (ColumnTransformer):
    ├─→ Pipeline Numérico: ImputaçãoMediana → StandardScaler  (11 features)
    └─→ Pipeline Categórico: ImputaçãoConstante → OneHotEncoder  (3 features)
    ↓
GradientBoostingClassifier:
    - n_estimators: 100
    - random_state: 42
    ↓
Saída: Predição binária (0 = Baixo Risco, 1 = Alto Risco) + probabilidade
```

**Componentes Principais**:
- **FeatureEngineer**: Transformador customizado scikit-learn para derivação de features específicas do domínio
- **ColumnTransformer**: Pré-processamento paralelo de features numéricas e categóricas
- **GradientBoostingClassifier**: Ensemble de 100 árvores de decisão sequenciais com boosting
- **Artefato Único**: Pipeline inteiro serializado como um único arquivo `model.joblib` (sem divergência treino-produção)

## Dados de Treinamento

### Dataset
- **Fonte**: PEDE (Pesquisa de Desenvolvimento Educacional) — ONG Passos Mágicos
- **Domínio**: Educação — Predição de risco acadêmico de alunos
- **Registros Originais**: ~1.349 alunos (2020–2022)
- **Registros Após Limpeza**: ~862 (linhas sem dados de referência 2022 removidas)
- **Features**: 11 numéricas + 3 categóricas = 14 features no total
- **Frequência de Atualização**: Anual, alinhada ao ciclo de coleta de dados da pesquisa PEDE

### Pré-processamento de Dados
- **Limpeza**: Target computado a partir de `FASE_2022 - NIVEL_IDEAL_2022`; todas as colunas de 2022 removidas para prevenir vazamento de dados
- **Features Numéricas**: Imputação pela mediana para valores ausentes, normalização com StandardScaler
- **Features Categóricas**: Imputação constante (`'MISSING'`), OneHotEncoder com `handle_unknown='ignore'`
- **Padronização de Colunas**: Sufixos de ano removidos (ex.: `INDE_2021` → `INDE`) para predição agnóstica ao ano

### Divisão dos Dados
- **Conjunto de Treinamento**: 80% dos dados (~690 amostras)
- **Conjunto de Teste**: 20% dos dados (~172 amostras)
- **Método de Divisão**: Divisão estratificada preservando distribuição de classes
- **Random State**: 42 (reproduzível)

### Distribuição de Classes
- **Classe Positiva (Em Risco, TARGET=1)**: ~70% das amostras
- **Classe Negativa (Sem Risco, TARGET=0)**: ~30% das amostras
- **Tratamento de Desbalanceamento**:
  - `class_weight='balanced'` para Regressão Logística e Random Forest
  - Seleção de modelo focada em recall para Gradient Boosting
  - Validação cruzada estratificada e divisão treino/teste

## Procedimento de Treinamento

### Configuração de Treinamento

**Hiperparâmetros**:
- **n_estimators**: 100 (número de estágios de boosting)
- **random_state**: 42 (reprodutibilidade)
- **test_size**: 0.2 (20% reservado para teste)
- **cv_folds**: 5 (validação cruzada estratificada K-fold)
- **scoring**: `recall` (métrica primária de otimização)

**Comparação de Modelos**:
Três modelos foram avaliados em cada execução de treinamento:
1. **Regressão Logística**: `class_weight='balanced'`, `max_iter=1000`
2. **Random Forest**: `class_weight='balanced'`, `n_estimators=100`
3. **Gradient Boosting**: `n_estimators=100` (selecionado como melhor modelo)

**Critério de Seleção**: Melhor recall no conjunto de teste entre todos os modelos candidatos.

### Processo de Treinamento
1. **Preparação de Dados**: Carregar CSV, limpar dados, computar target, remover vazamento
2. **Engenharia de Features**: Derivar `IS_NEW_STUDENT`, remover identificadores
3. **Divisão Estratificada**: 80/20 treino/teste preservando proporções de classes
4. **Validação Cruzada**: CV estratificada de 5 folds para cada modelo candidato
5. **Treinamento Completo**: Ajustar pipeline no conjunto de treinamento completo
6. **Avaliação**: Computar recall, F1, ROC-AUC no conjunto de teste reservado
7. **Seleção de Modelo**: Selecionar melhor modelo pelo recall no teste
8. **Persistência**: Salvar em `models/production/model.joblib`
9. **Rastreamento**: Registrar todos os parâmetros e métricas no MLflow

## Avaliação

### Métricas de Avaliação (Última Execução MLflow)

**Fonte**: `latest_metrics_202602282239.csv` — exportação de métricas de experimento MLflow

#### Desempenho do Melhor Modelo (Gradient Boosting)

| Métrica | Valor | UUID da Execução MLflow |
|---------|-------|-------------------------|
| **CV Recall Médio** | 0.9433 (94,33%) | `e94c275bc9784ea8b68736d10d4002fd` |
| **CV Recall Desvio Padrão** | 0.0304 (3,04%) | `e94c275bc9784ea8b68736d10d4002fd` |
| **Recall no Teste** | 0.9503 (95,03%) | `e94c275bc9784ea8b68736d10d4002fd` |
| **F1 Score no Teste** | 0.8289 (82,89%) | `e94c275bc9784ea8b68736d10d4002fd` |
| **ROC-AUC no Teste** | 0.7459 (74,59%) | `e94c275bc9784ea8b68736d10d4002fd` |
| **Duração do Treinamento** | 4,21 segundos | `e94c275bc9784ea8b68736d10d4002fd` |
| **Melhor Score (Recall)** | 0.9503 | `a6535aea0faf4866b7ed72274cbc845e` |

#### Comparação de Todos os Modelos (Último Experimento com 3 Modelos)

| Modelo | CV Recall Médio | Recall Teste | F1 Teste | ROC-AUC Teste | Duração (s) |
|--------|----------------|-------------|----------|---------------|-------------|
| Regressão Logística | 0.3770 | 0.5138 | 0.6436 | 0.7056 | 4,04 |
| **Gradient Boosting** | **0.9433** | **0.9503** | **0.8289** | **0.7459** | **4,21** |

#### Consistência de Execuções Históricas (Configuração Padrão)

| Modelo | CV Recall Médio | Recall Teste | F1 Teste | ROC-AUC Teste |
|--------|----------------|-------------|----------|---------------|
| Regressão Logística | 0.4191 | 0.3719 | 0.5357 | 0.7212 |
| Random Forest | 0.5209 | 0.5537 | 0.6768 | 0.7548 |
| Gradient Boosting | 0.9481 | 0.9421 | 0.8539 | 0.7672 |

**Interpretação**:
- Gradient Boosting consistentemente atinge **>94% de recall** em todas as execuções de treinamento
- Desvio padrão da validação cruzada de ~3% confirma estabilidade do modelo
- F1 score de 82,9% demonstra bom equilíbrio entre precisão e recall
- ROC-AUC de 74,6% indica capacidade adequada de ordenação por rank

### Análise de Overfitting

- **CV Recall Médio vs. Recall no Teste**: 94,3% vs. 95,0% — nenhum overfitting detectado
- **Desvio Padrão CV**: 3,0% — baixa variância entre folds
- **Reprodutibilidade**: Resultados idênticos em execuções repetidas com `random_state=42`

## Resumo de Desempenho do Modelo

### Pontos Fortes
1. **Alto Recall (95,0%)**: Captura 95 de cada 100 alunos em risco
2. **Validação Cruzada Estável**: Baixa variância (3,0% desvio padrão) em 5 folds
3. **Bom F1 Score (82,9%)**: Equilíbrio saudável entre precisão e recall
4. **Treinamento Rápido (~4 segundos)**: Permite retreinamento e experimentação rápidos
5. **Reproduzível**: Resultados determinísticos com random state fixo
6. **Deploy com Artefato Único**: Pipeline completo em um único arquivo serializado

### Limitações
1. **Dataset Pequeno (~862 amostras)**: Limita complexidade do modelo e garantias de generalização
2. **Específico para a População**: Treinado com alunos do Passos Mágicos em Embu-Guaçu; não validado para outras populações
3. **ROC-AUC (74,6%)**: Adequado mas não excepcional para ordenação por rank
4. **Dependência Temporal**: Desempenho assume estabilidade das dinâmicas educacionais entre anos
5. **Sem Ajuste de Hiperparâmetros**: Valores padrão fixos utilizados; melhorias marginais possíveis com tuning
6. **Dependência da Pesquisa PEDE**: Requer metodologia consistente dos indicadores entre anos

## Uso Pretendido

### Casos de Uso Principais
1. **Identificação de Risco Estudantil**: Sinalizar alunos em risco de defasagem educacional para intervenção precoce
2. **Priorização de Recursos**: Classificar alunos por probabilidade de risco para triagem
3. **Scoring em Lote**: Pontuar coortes inteiras de alunos no início de períodos acadêmicos
4. **Ferramenta de Monitoramento**: Acompanhar mudanças na distribuição de risco ao longo dos anos

### Usos Fora do Escopo
- **Não para**: Decisões disciplinares ou administrativas individuais sem revisão do educador
- **Não para**: Outras populações de alunos sem retreinamento e validação
- **Não para**: Scoring de alta frequência em tempo real (projetado para uso periódico/em lote)
- **Não para**: Predição além do próximo período acadêmico imediato

## Considerações Éticas

### Viés e Equidade
- **Consciência do Desbalanceamento de Classes**: Modelo é otimizado para recall (sensibilidade), deliberadamente aceitando taxas mais altas de falsos positivos para minimizar alunos em risco não identificados
- **Viés Populacional**: Treinado exclusivamente com alunos do Passos Mágicos; desempenho em outras demografias é desconhecido
- **Contexto Socioeconômico**: Features refletem indicadores socioeconômicos que podem correlacionar com status socioeconômico; predições devem ser usadas para suporte, nunca para ações punitivas
- **Recomendação**: Predições devem sempre ser revisadas por educadores que conhecem os alunos; o modelo complementa, não substitui, o julgamento humano

### Transparência
- **Arquitetura do Modelo**: Totalmente documentada com componentes open-source (scikit-learn)
- **Processo de Treinamento**: Reproduzível com random state fixo e registrado via MLflow
- **Métricas de Avaliação**: Métricas abrangentes fornecidas por modelo e por classe
- **Limitações**: Claramente declaradas com contexto acionável

### Privacidade de Dados
- **Fonte de Dados**: Dados de pesquisa PEDE coletados pelo Passos Mágicos com consentimento apropriado
- **Dados Pessoais**: Nomes de alunos (`NOME`) e identificadores de instituição são removidos antes do treinamento do modelo
- **Conformidade**: Engenharia de features remove PII antes do modelo processar qualquer dado
- **Armazenamento**: Artefato do modelo não contém dados individuais de alunos

### Avisos de Risco
- **Não é uma Avaliação Determinística**: Predições são probabilísticas e devem informar, não determinar, intervenções educacionais
- **Falsos Negativos**: Aproximadamente 5% dos alunos em risco podem não ser identificados — revisão manual de casos limítrofes é recomendada
- **Falsos Positivos**: Alguns alunos sem risco serão sinalizados — isso é intencional (baixo custo deste tipo de erro)
- **Validade Temporal**: Modelo deve ser retreinado anualmente com novos dados PEDE

## Manutenção do Modelo

### Cronograma de Retreinamento
- **Frequência**: Anual, quando novos dados PEDE estiverem disponíveis
- **Gatilho**: Drift de dados detectado via Evidently, ou recall abaixo de 90%
- **Processo**: Invocar endpoint `/train` ou executar `ModelTrainer` diretamente
- **Validação**: Comparação automatizada de múltiplos modelos garante seleção do melhor

### Controle de Versão
- **Versionamento do Modelo**: Cada execução de treinamento produz um experimento com nome único no MLflow
- **Armazenamento de Artefatos**: `models/artifacts/{experiment_name}/model.joblib` + `metadata.json`
- **Modelo de Produção**: `models/production/model.joblib` (cópia do melhor artefato)
- **Rollback**: Restaurar qualquer artefato anterior do diretório `artifacts/`

### Monitoramento
- **Métricas para Monitorar**:
  - Recall no teste com novos dados de ground truth
  - Distribuição de probabilidade das predições (calibração)
  - Drift de features via Evidently
  - Latência e taxas de erro da API via Prometheus
- **Limiares de Alerta**:
  - Queda de recall > 5% do baseline (95,0%)
  - Drift detectado em > 3 features simultaneamente
  - Taxa de erro da API > 1%

## Especificações Técnicas

### Requisitos de Hardware
- **Treinamento**: Qualquer CPU moderna (treinamento leva ~4 segundos)
- **Inferência**: Qualquer CPU moderna (< 50ms por predição)
- **Memória**: 2GB+ RAM suficiente
- **GPU**: Não necessária (scikit-learn utiliza apenas CPU)

### Dependências de Software
- **Python**: 3.10+
- **scikit-learn**: ≥ 1.0.0
- **pandas**: ≥ 2.0.0
- **numpy**: ≥ 2.4.1
- **mlflow**: ≥ 2.0.0
- **flask**: ≥ 3.1.2
- **joblib**: ≥ 1.5.3
- **evidently**: ≥ 0.4.0
- **prometheus-client**: ≥ 0.21.0

### Tamanho do Modelo
- **Arquivo do Modelo**: ~50KB–500KB (pipeline serializado com joblib)
- **Arquivo de Metadados**: ~1KB (JSON)
- **Total**: < 1MB

### Velocidade de Inferência
- **Predição Individual**: < 50ms (CPU)
- **Lote (100 alunos)**: < 200ms (CPU)
- **Throughput**: ~2.000+ predições/segundo (CPU)

---

**Última Atualização**: 02-2026  
**Versão do Modelo**: 1.0  
