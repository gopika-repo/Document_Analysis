from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.onprem.client import Users
from diagrams.onprem.database import PostgreSQL, Redis
from diagrams.onprem.container import Docker
from diagrams.onprem.inmemory import Redis as RedisMem
from diagrams.onprem.network import Nginx
from diagrams.onprem.queue import Kafka
from diagrams.programming.framework import FastAPI
from diagrams.generic.storage import Storage
from diagrams.aws.ml import Sagemaker
from diagrams.generic.device import Mobile

graph_attr = {
    "fontsize": "20",
    "bgcolor": "white"
}

with Diagram("Vision-Fusion Architecture", show=False, direction="LR", graph_attr=graph_attr):
    users = Users("Users")
    
    with Cluster("Frontend Layer"):
        api_gateway = Nginx("API Gateway")
        web_ui = Mobile("Web UI")
    
    with Cluster("API Layer"):
        fastapi = FastAPI("FastAPI Server")
        
    with Cluster("Agent System"):
        with Cluster("Multi-Agent Orchestrator"):
            orchestrator = Python("Agent Orchestrator")
            
        with Cluster("Specialized Agents"):
            vision = Python("Vision Agent")
            text = Python("Text Agent")
            fusion = Python("Fusion Agent")
            validation = Python("Validation Agent")
        
        orchestrator >> vision
        orchestrator >> text
        orchestrator >> fusion
        orchestrator >> validation
    
    with Cluster("CV & OCR Processing"):
        yolo = Sagemaker("YOLOv8")
        ocr = Python("OCR Engine")
        layout = Python("Layout Analyzer")
    
    with Cluster("RAG System"):
        qdrant = Storage("Qdrant Vector DB")
        embeddings = Python("Embedding Engine")
        retriever = Python("Multi-Modal Retriever")
    
    with Cluster("Services"):
        pdf = Python("PDF Processor")
        image = Python("Image Processor")
        confidence = Python("Confidence Engine")
    
    with Cluster("Storage"):
        uploads = Storage("Document Storage")
        cache = RedisMem("Redis Cache")
        metadata = PostgreSQL("Metadata DB")
    
    # Connections
    users >> api_gateway >> fastapi
    web_ui >> api_gateway
    
    fastapi >> orchestrator
    fastapi >> uploads
    
    orchestrator >> pdf
    orchestrator >> image
    
    pdf >> yolo
    pdf >> ocr
    
    image >> yolo
    image >> ocr
    
    yolo >> layout
    ocr >> text
    
    layout >> vision
    text >> fusion
    vision >> fusion
    
    fusion >> validation
    validation >> confidence
    
    fusion >> embeddings >> qdrant
    confidence >> metadata
    
    qdrant >> retriever >> fastapi
    
    # Cache connections
    fastapi >> cache
    retriever >> cache

print("Diagram generated as architecture.png")