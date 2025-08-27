graph TD

    A[CLI Input] --> B[Command Parser]
    B --> ZZ[Language Model]
    ZZ --> C{Task Selection}
    
    C -->|summarize| D[Summarization Module]
    C -->|explain| E[Explanation Module]
    C -->|classify| F[Classification Module]
    
    D --> G[PDF Parser]
    E --> G
   
    G --> H[Text Preprocessing]
    H --> I[Text Chunking]
    
    %% Summarization Path
    I --> V[Language Model]
    
    %% Explanation Path
    I --> M[Embedding Model]
    E --> N[Query Processor]
    N --> M
    
    M --> N1[Query Embedding]
    N1 --> O[Semantic Search]
    M --> P[(Vector Database)]
    O --> P
    P --> Q[Context Retrieval]
    
    %% Classification Path
    F --> S[Folder Parser]
    S --> SS[Extract Metadata for each existing paper]
    SS --> M
    
    %% New paper classification
    F --> BB[Extract New Paper Metadata]
    BB --> M
    
    P --> U{Confidence > Threshold?}
    
    %% Language Model Integration
    Q --> V[Language Model]
    V --> W[Response Generator]
    
    %% Output Processing
    W --> X[Output Formatter]
    
    %% Classification Output
    U -->|Yes| Y[Assign to Similar Papers' Folder]
    U -->|No| Z[Create New Folder]
    Y --> X
    Z --> X
    
    X --> AA[Terminal Output]
