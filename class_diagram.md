# BiztelAI Data Pipeline - Class Diagram

```mermaid
classDiagram
    %% Abstract base class
    class DataProcessor {
        <<abstract>>
        +name: str
        +config: Dict
        +metadata: Dict
        +process(data, **kwargs)* DataFrame
        +get_metadata() Dict
        +add_metadata(key, value) None
        +validate(data) bool
        +with_timing() DataProcessor
        +with_logging() DataProcessor
        +from_config(config) DataProcessor
    }
    
    %% Decorators
    class ProcessorDecorator {
        +processor: DataProcessor
        +process(data, **kwargs) DataFrame
        +get_metadata() Dict
        +validate(data) bool
    }
    
    class TimingDecorator {
        +process(data, **kwargs) DataFrame
    }
    
    class LoggingDecorator {
        +process(data, **kwargs) DataFrame
    }
    
    %% Composite processor
    class CompositeProcessor {
        +processors: List[DataProcessor]
        +add_processor(processor) CompositeProcessor
        +process(data, **kwargs) DataFrame
        +get_processor(name) DataProcessor
    }
    
    %% Factory
    class ProcessorFactory {
        <<static>>
        +create_processor(processor_type, config) DataProcessor
        +from_config_file(config_file) CompositeProcessor
    }
    
    %% Concrete processors
    class DataLoader {
        +file_path: str
        +process(data, **kwargs) DataFrame
        +load_json(file_path) DataFrame
    }
    
    class DataCleaner {
        +process(data, **kwargs) DataFrame
        +handle_missing_values(df) DataFrame
        +remove_duplicates(df) DataFrame
        +clean_text_data(df) DataFrame
        +fix_data_types(df) DataFrame
    }
    
    class TextPreprocessor {
        +text_column: str
        +process(data, **kwargs) DataFrame
        +tokenize(text) List[str]
        +remove_stopwords(tokens) List[str]
        +lemmatize(tokens) List[str]
    }
    
    class FeatureTransformer {
        +process(data, **kwargs) DataFrame
        +transform_categorical(df) DataFrame
        +create_text_features(df) DataFrame
    }
    
    %% Relationships
    DataProcessor <|-- CompositeProcessor
    DataProcessor <|-- DataLoader
    DataProcessor <|-- DataCleaner
    DataProcessor <|-- TextPreprocessor
    DataProcessor <|-- FeatureTransformer
    
    ProcessorDecorator *-- DataProcessor
    ProcessorDecorator <|-- TimingDecorator
    ProcessorDecorator <|-- LoggingDecorator
    
    CompositeProcessor o-- DataProcessor : contains
    
    note for ProcessorFactory "Creates and configures processors"
    note for CompositeProcessor "Implements composite pattern"
    note for ProcessorDecorator "Implements decorator pattern"
```

## Design Patterns Used

1. **Abstract Factory Pattern** - `ProcessorFactory` creates different types of processors
2. **Composite Pattern** - `CompositeProcessor` allows treating individual processors and compositions of processors uniformly
3. **Decorator Pattern** - `ProcessorDecorator` and its subclasses add functionality to processors without modifying their code
4. **Template Method Pattern** - `DataProcessor` defines the skeleton of the processing algorithm, deferring specific steps to subclasses

## Pipeline Configuration

The pipeline is configured using a JSON configuration file that specifies:
- Pipeline metadata (name, description, version)
- Input and output files
- Logging configuration
- Processor chain with processor-specific configurations
- Caching and parallelism options

## Runtime Flow

1. `run_pipeline.py` parses command-line arguments and loads configuration
2. `ProcessorFactory` creates a pipeline from the configuration
3. Pipeline processes the data through each processor in sequence
4. Each processor applies its specific transformation to the data
5. Decorators add timing and logging functionality around processor execution
6. Results are written to the specified output file

## Benefits of the Architecture

1. **Modularity**: Each processor handles a specific aspect of data processing
2. **Extensibility**: New processors can be easily added by subclassing `DataProcessor`
3. **Configurability**: Pipeline behavior can be modified through configuration without code changes
4. **Observability**: Timing and logging decorators provide insights into processing performance
5. **Reusability**: Processors can be reused in different pipelines and configurations 