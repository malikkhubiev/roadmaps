Да, MLflow и LlamaIndex вместе могут создать мощную систему контроля версий для LLM-цепочки. Вот как это можно организовать:

## Архитектура системы

**MLflow отвечает за:**
- Версионирование промптов (YAML/JSON)
- Трекинг экспериментов и метрик
- Регистрация моделей и промптов
- Сравнительный анализ LLM
- Дашборды и визуализации

**LlamaIndex отвечает за:**
- Оркестрация цепочек LLM
- Управление различными LLM провайдерами
- Векторизация и эмбеддинги
- Абстракция над разными API

## Реализация системы

### 1. Конфигурация (YAML промпты)

```yaml
# prompts/pipeline_config.yaml
version: 1.0
chain:
  - name: data_extractor
    llm_providers:
      - deepseek
      - yandexgpt
      - gigachat
    prompt: |
      Извлеки ключевые данные из текста:
      {context}
      
      Структура:
      1. Основная тема
      2. Ключевые факты
      3. Выводы
      
  - name: summarizer
    llm_providers:
      - deepseek
      - yandexgpt
    prompt: |
      Суммаризируй текст:
      {extracted_data}
      
      Требования:
      - Максимум 3 предложения
      - Сохрани ключевую информацию
      
  - name: analyzer
    llm_providers:
      - gigachat
      - yandexgpt
    prompt: |
      Проанализируй текст:
      {summary}
      
      Аспекты анализа:
      1. Тональность
      2. Достоверность
      3. Практическая ценность
```

### 2. MLflow для контроля версий и трекинга

```python
# mlflow_tracker.py
import mlflow
import yaml
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import json

class PromptVersioner:
    def __init__(self, tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_name = "llm_chain_versions"
        mlflow.set_experiment(self.experiment_name)
    
    def log_prompt_version(self, 
                          prompt_name: str, 
                          prompt_content: str,
                          prompt_type: str = "system",
                          metadata: Dict = None):
        """Логирование версии промпта"""
        with mlflow.start_run(run_name=f"prompt_{prompt_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Логируем промпт как артефакт
            with open(f"{prompt_name}.yaml", "w") as f:
                f.write(prompt_content)
            
            mlflow.log_artifact(f"{prompt_name}.yaml", "prompts")
            
            # Логируем метаданные
            if metadata:
                mlflow.log_params(metadata)
                mlflow.log_dict(metadata, "metadata.json")
            
            # Регистрируем промпт как модель
            mlflow.pyfunc.log_model(
                artifact_path="prompt_model",
                python_model=None,
                artifacts={f"{prompt_name}.yaml": f"{prompt_name}.yaml"},
                registered_model_name=f"prompt_{prompt_name}"
            )
    
    def log_llm_performance(self,
                           chain_step: str,
                           llm_provider: str,
                           metrics: Dict[str, float],
                           input_data: str = None,
                           output_data: str = None):
        """Логирование производительности LLM"""
        with mlflow.start_run(run_name=f"{chain_step}_{llm_provider}_{datetime.now().strftime('%H%M%S')}", nested=True):
            mlflow.log_params({
                "chain_step": chain_step,
                "llm_provider": llm_provider,
                "timestamp": datetime.now().isoformat()
            })
            
            mlflow.log_metrics(metrics)
            
            if input_data:
                mlflow.log_text(input_data, "input.txt")
            
            if output_data:
                mlflow.log_text(output_data, "output.txt")
    
    def compare_embeddings(self,
                          embedding_models: List[str],
                          test_texts: List[str],
                          metrics: Dict[str, Dict[str, float]]):
        """Сравнение моделей эмбеддингов"""
        with mlflow.start_run(run_name="embedding_comparison"):
            for model_name, model_metrics in metrics.items():
                mlflow.log_metrics(
                    {f"{model_name}_{k}": v for k, v in model_metrics.items()}
                )
            
            # Логируем результаты сравнения
            comparison_df = pd.DataFrame(metrics).T
            mlflow.log_table(comparison_df, "embedding_comparison.json")
```

### 3. LlamaIndex для оркестрации цепочек

```python
# llm_orchestrator.py
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from typing import Dict, List, Any, Optional
import asyncio
import time

class LLMOrchestrator:
    def __init__(self, mlflow_tracker):
        self.mlflow = mlflow_tracker
        self.llm_providers = {
            "deepseek": {
                "class": OpenAI,
                "config": {
                    "api_key": "your_deepseek_key",
                    "base_url": "https://api.deepseek.com/v1",
                    "model": "deepseek-chat"
                }
            },
            "yandexgpt": {
                "class": OpenAI,
                "config": {
                    "api_key": "your_yandex_key",
                    "base_url": "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                    "model": "yandexgpt"
                }
            },
            "gigachat": {
                "class": OpenAI,
                "config": {
                    "api_key": "your_gigachat_key",
                    "base_url": "https://gigachat.devices.sberbank.ru/api/v1",
                    "model": "GigaChat"
                }
            }
        }
        
        self.embedding_models = {
            "openai": OpenAIEmbedding(model="text-embedding-3-small"),
            "minilm": HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            "multilingual": HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
        }
    
    async def execute_chain_step(self,
                                step_name: str,
                                prompt_template: str,
                                providers: List[str],
                                context: str,
                                **kwargs) -> Dict[str, Any]:
        """Выполнение шага цепочки с несколькими LLM"""
        results = {}
        metrics = {}
        
        for provider in providers:
            try:
                start_time = time.time()
                
                # Инициализируем LLM
                llm_config = self.llm_providers[provider]
                llm = llm_config["class"](**llm_config["config"])
                
                # Создаем промпт
                prompt = PromptTemplate(prompt_template)
                formatted_prompt = prompt.format(context=context, **kwargs)
                
                # Выполняем запрос
                response = await llm.acomplete(formatted_prompt)
                end_time = time.time()
                
                # Собираем метрики
                latency = end_time - start_time
                token_count = len(response.text.split())
                
                step_metrics = {
                    "latency": latency,
                    "token_count": token_count,
                    "response_length": len(response.text),
                    "success": True
                }
                
                results[provider] = {
                    "response": response.text,
                    "metrics": step_metrics
                }
                
                # Логируем в MLflow
                self.mlflow.log_llm_performance(
                    chain_step=step_name,
                    llm_provider=provider,
                    metrics=step_metrics,
                    input_data=formatted_prompt,
                    output_data=response.text
                )
                
            except Exception as e:
                error_metrics = {
                    "success": False,
                    "error": str(e),
                    "latency": 0,
                    "token_count": 0
                }
                results[provider] = {
                    "response": None,
                    "metrics": error_metrics
                }
                self.mlflow.log_llm_performance(
                    chain_step=step_name,
                    llm_provider=provider,
                    metrics=error_metrics
                )
        
        return results
    
    def compare_embedding_quality(self,
                                 texts: List[str],
                                 reference_embeddings: Optional[List] = None) -> Dict:
        """Сравнение качества эмбеддингов"""
        results = {}
        
        for model_name, embed_model in self.embedding_models.items():
            try:
                # Генерация эмбеддингов
                embeddings = embed_model.get_text_embedding_batch(texts)
                
                # Вычисление метрик
                metrics = self._calculate_embedding_metrics(
                    embeddings, 
                    reference_embeddings
                )
                
                results[model_name] = metrics
                
            except Exception as e:
                results[model_name] = {"error": str(e)}
        
        # Логируем сравнение в MLflow
        self.mlflow.compare_embeddings(
            list(self.embedding_models.keys()),
            texts,
            results
        )
        
        return results
    
    def _calculate_embedding_metrics(self, 
                                    embeddings: List,
                                    reference: Optional[List] = None) -> Dict:
        """Вычисление метрик качества эмбеддингов"""
        import numpy as np
        
        metrics = {}
        emb_array = np.array(embeddings)
        
        # 1. Косинусное сходство внутри набора
        similarities = []
        for i in range(len(emb_array)):
            for j in range(i+1, len(emb_array)):
                sim = np.dot(emb_array[i], emb_array[j]) / (
                    np.linalg.norm(emb_array[i]) * np.linalg.norm(emb_array[j])
                )
                similarities.append(sim)
        
        metrics["mean_similarity"] = np.mean(similarities)
        metrics["std_similarity"] = np.std(similarities)
        
        # 2. Норма векторов
        metrics["mean_norm"] = np.mean([np.linalg.norm(e) for e in emb_array])
        metrics["std_norm"] = np.std([np.linalg.norm(e) for e in emb_array])
        
        return metrics

class ChainExecutor:
    def __init__(self, orchestrator: LLMOrchestrator):
        self.orchestrator = orchestrator
        self.chain_history = []
    
    async def execute_full_chain(self, 
                                chain_config: Dict,
                                input_data: str) -> Dict:
        """Выполнение полной цепочки"""
        chain_results = {}
        current_context = input_data
        
        for step in chain_config["chain"]:
            step_name = step["name"]
            providers = step["llm_providers"]
            prompt_template = step["prompt"]
            
            # Параллельное выполнение на всех провайдерах
            step_results = await self.orchestrator.execute_chain_step(
                step_name=step_name,
                prompt_template=prompt_template,
                providers=providers,
                context=current_context
            )
            
            # Выбор лучшего результата (можно настроить логику)
            best_provider = self._select_best_provider(step_results)
            best_response = step_results[best_provider]["response"]
            
            chain_results[step_name] = {
                "all_results": step_results,
                "best_provider": best_provider,
                "best_response": best_response
            }
            
            # Обновляем контекст для следующего шага
            current_context = best_response
        
        return chain_results
    
    def _select_best_provider(self, step_results: Dict) -> str:
        """Выбор лучшего провайдера на основе метрик"""
        # Пример: выбираем по комбинации latency и token_count
        scores = {}
        for provider, data in step_results.items():
            metrics = data["metrics"]
            if metrics.get("success", False):
                # Чем меньше latency и больше token_count, тем лучше
                score = metrics.get("token_count", 0) / max(metrics.get("latency", 1), 0.1)
                scores[provider] = score
        
        return max(scores, key=scores.get) if scores else list(step_results.keys())[0]
```

### 4. Основной пайплайн

```python
# main_pipeline.py
import yaml
import asyncio
from datetime import datetime

async def main():
    # Инициализация трекера
    tracker = PromptVersioner()
    
    # Загрузка конфигурации цепочки
    with open("prompts/pipeline_config.yaml", "r") as f:
        chain_config = yaml.safe_load(f)
    
    # Логирование версии конфигурации
    tracker.log_prompt_version(
        prompt_name="pipeline_config",
        prompt_content=yaml.dump(chain_config),
        prompt_type="chain_config",
        metadata={
            "version": chain_config.get("version", "1.0"),
            "steps": len(chain_config.get("chain", [])),
            "timestamp": datetime.now().isoformat()
        }
    )
    
    # Инициализация оркестратора
    orchestrator = LLMOrchestrator(tracker)
    executor = ChainExecutor(orchestrator)
    
    # Тестовые данные
    test_text = """
    Искусственный интеллект совершил прорыв в медицинской диагностике...
    """
    
    # Выполнение цепочки
    chain_results = await executor.execute_full_chain(
        chain_config=chain_config,
        input_data=test_text
    )
    
    # Сравнение эмбеддингов
    embedding_results = orchestrator.compare_embedding_quality(
        texts=[test_text, "Другой пример текста для сравнения"],
        reference_embeddings=None
    )
    
    # Генерация отчета
    generate_report(chain_results, embedding_results)

def generate_report(chain_results: Dict, embedding_results: Dict):
    """Генерация сравнительного отчета"""
    report = {
        "chain_performance": {},
        "embedding_comparison": embedding_results,
        "summary": {}
    }
    
    for step_name, step_data in chain_results.items():
        report["chain_performance"][step_name] = {
            "best_provider": step_data["best_provider"],
            "all_metrics": {
                provider: data["metrics"]
                for provider, data in step_data["all_results"].items()
            }
        }
    
    # Сохранение отчета
    import json
    with open(f"reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("Отчет сгенерирован и сохранен")

if __name__ == "__main__":
    asyncio.run(main())
```

### 5. Дашборд для мониторинга

```python
# dashboard.py
import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st

def create_dashboard():
    st.title("LLM Chain Monitoring Dashboard")
    
    # Подключение к MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # 1. Сравнение производительности LLM
    st.header("Сравнение LLM провайдеров")
    
    # Получение данных из MLflow
    experiments = mlflow.search_experiments()
    
    for exp in experiments:
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        
        if not runs.empty:
            # Визуализация latency по провайдерам
            fig = px.bar(
                runs,
                x='params.llm_provider',
                y='metrics.latency',
                color='params.chain_step',
                title=f"Latency по провайдерам ({exp.name})"
            )
            st.plotly_chart(fig)
    
    # 2. Качество эмбеддингов
    st.header("Сравнение моделей эмбеддингов")
    
    # 3. История версий промптов
    st.header("Версионирование промптов")
    
    # Получение зарегистрированных моделей (промптов)
    client = mlflow.MlflowClient()
    registered_models = client.search_registered_models()
    
    for model in registered_models:
        if model.name.startswith("prompt_"):
            st.subheader(model.name)
            
            # Показать версии
            versions = client.search_model_versions(f"name='{model.name}'")
            for v in versions:
                st.write(f"Версия {v.version}: {v.description}")
```

## Структура проекта

```
llm-chain-versioning/
├── prompts/
│   ├── pipeline_config.yaml
│   ├── system_prompts/
│   └── user_prompts/
├── mlflow_tracker.py
├── llm_orchestrator.py
├── chain_executor.py
├── main_pipeline.py
├── dashboard.py
├── tests/
└── reports/
```

## Ключевые преимущества системы:

1. **Полное версионирование**: Все промпты и конфигурации хранятся в MLflow
2. **Real-time мониторинг**: Статистика по каждому LLM в реальном времени
3. **Сравнительный анализ**: Автоматическое сравнение разных провайдеров
4. **Гибкость**: Легко добавлять новых провайдеров и шаги цепочки
5. **Визуализация**: Дашборды для анализа производительности
6. **Качество эмбеддингов**: Метрики для сравнения разных моделей векторизации

Система позволяет в реальном времени видеть, какой LLM лучше справляется на каждом этапе, отслеживать изменения промптов и сравнивать качество разных подходов к векторизации.
