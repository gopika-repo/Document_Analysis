import pytest
import numpy as np
import asyncio
from app.agents.vision_agent import VisionAgent, VisionState
from app.agents.text_agent import TextAgent, TextState
from app.agents.fusion_agent import FusionAgent, FusionState
from app.agents.validation_agent import ValidationAgent, ValidationState
from app.utils.error_handler import AgentException

class TestVisionAgent:
    """Test Vision Agent"""
    
    def setup_method(self):
        self.vision_agent = VisionAgent()
    
    def create_test_images(self):
        """Create test images for processing"""
        images = []
        for i in range(2):
            img = np.ones((300, 400, 3), dtype=np.uint8) * 255
            # Add some text
            import cv2
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f'Test Page {i+1}', (50, 150), font, 1, (0, 0, 0), 2)
            images.append(img)
        return images
    
    def test_initialization(self):
        """Test vision agent initialization"""
        assert self.vision_agent is not None
        assert hasattr(self.vision_agent, 'yolo_model')
        assert hasattr(self.vision_agent, 'layout_analyzer')
    
    def test_graph_creation(self):
        """Test LangGraph creation"""
        graph = self.vision_agent.create_graph()
        
        assert graph is not None
        assert hasattr(graph, 'nodes')
        assert hasattr(graph, 'edges')
    
    def test_state_validation(self):
        """Test vision state validation"""
        state = VisionState(
            document_id="test_doc_123",
            images=self.create_test_images()
        )
        
        assert state.document_id == "test_doc_123"
        assert len(state.images) == 2
        assert len(state.errors) == 0
    
    @pytest.mark.asyncio
    async def test_document_processing(self):
        """Test complete document processing"""
        test_images = self.create_test_images()
        
        result = await self.vision_agent.process_document(
            document_id="test_doc_123",
            images=test_images
        )
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "document_id" in result
        assert result["document_id"] == "test_doc_123"
        
        if result["success"]:
            assert "visual_summary" in result
            assert "detections" in result
    
    def test_detection_logic(self):
        """Test element detection logic"""
        state = VisionState(
            document_id="test_doc",
            images=self.create_test_images()
        )
        
        updated_state = self.vision_agent.detect_elements(state)
        
        assert isinstance(updated_state, VisionState)
        assert len(updated_state.detections) > 0 or len(updated_state.errors) > 0
    
    def test_layout_analysis_logic(self):
        """Test layout analysis logic"""
        state = VisionState(
            document_id="test_doc",
            images=self.create_test_images()
        )
        
        # First run detection
        state = self.vision_agent.detect_elements(state)
        
        # Then run layout analysis
        updated_state = self.vision_agent.analyze_layout(state)
        
        assert isinstance(updated_state, VisionState)
        assert len(updated_state.layout_analysis) > 0 or len(updated_state.errors) > 0
    
    def test_summary_generation(self):
        """Test visual summary generation"""
        state = VisionState(
            document_id="test_doc",
            images=self.create_test_images()
        )
        
        # Run through pipeline
        state = self.vision_agent.detect_elements(state)
        state = self.vision_agent.analyze_layout(state)
        updated_state = self.vision_agent.generate_summary(state)
        
        assert isinstance(updated_state, VisionState)
        assert "visual_summary" in updated_state.dict()
        
        summary = updated_state.visual_summary
        assert "document_id" in summary
        assert "total_pages" in summary
        assert "detection_summary" in summary

class TestTextAgent:
    """Test Text Agent"""
    
    def setup_method(self):
        self.text_agent = TextAgent()
    
    def create_test_images(self):
        """Create test images with text"""
        images = []
        img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        
        # Add text to image
        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Invoice No: INV-2024-001', (50, 100), font, 0.7, (0, 0, 0), 2)
        cv2.putText(img, 'Date: 2024-01-15', (50, 140), font, 0.7, (0, 0, 0), 2)
        cv2.putText(img, 'Amount: $1,250.00', (50, 180), font, 0.7, (0, 0, 0), 2)
        cv2.putText(img, 'Vendor: ABC Corporation', (50, 220), font, 0.7, (0, 0, 0), 2)
        
        images.append(img)
        return images
    
    def test_initialization(self):
        """Test text agent initialization"""
        assert self.text_agent is not None
        assert hasattr(self.text_agent, 'ocr_engine')
        assert hasattr(self.text_agent, 'llm_client')
    
    def test_graph_creation(self):
        """Test LangGraph creation"""
        graph = self.text_agent.create_graph()
        
        assert graph is not None
        assert hasattr(graph, 'nodes')
        assert hasattr(graph, 'edges')
    
    @pytest.mark.asyncio
    async def test_document_processing(self):
        """Test complete document processing"""
        test_images = self.create_test_images()
        
        result = await self.text_agent.process_document(
            document_id="test_doc_123",
            images=test_images
        )
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "document_id" in result
        
        if result["success"]:
            assert "text_summary" in result
            assert "extracted_entities" in result
    
    def test_ocr_logic(self):
        """Test OCR logic"""
        state = TextState(
            document_id="test_doc",
            images=self.create_test_images()
        )
        
        updated_state = self.text_agent.perform_ocr(state)
        
        assert isinstance(updated_state, TextState)
        assert len(updated_state.ocr_results) > 0 or len(updated_state.errors) > 0
    
    def test_entity_extraction_logic(self):
        """Test entity extraction logic"""
        state = TextState(
            document_id="test_doc",
            images=self.create_test_images()
        )
        
        # First run OCR
        state = self.text_agent.perform_ocr(state)
        
        # Then extract entities
        updated_state = self.text_agent.extract_entities(state)
        
        assert isinstance(updated_state, TextState)
        assert len(updated_state.extracted_entities) > 0 or len(updated_state.errors) > 0
        
        # Check entity categories
        entities = updated_state.extracted_entities
        expected_categories = ["dates", "amounts", "names", "organizations", "locations", "other_fields"]
        assert any(cat in entities for cat in expected_categories)
    
    def test_semantic_analysis_logic(self):
        """Test semantic analysis logic"""
        state = TextState(
            document_id="test_doc",
            images=self.create_test_images()
        )
        
        # Run through pipeline
        state = self.text_agent.perform_ocr(state)
        state = self.text_agent.extract_entities(state)
        updated_state = self.text_agent.analyze_semantics(state)
        
        assert isinstance(updated_state, TextState)
        assert "semantic_analysis" in updated_state.dict()

class TestFusionAgent:
    """Test Fusion Agent"""
    
    def setup_method(self):
        self.fusion_agent = FusionAgent()
    
    def create_test_vision_results(self):
        """Create test vision results"""
        return {
            "success": True,
            "document_id": "test_doc_123",
            "visual_summary": {
                "document_id": "test_doc_123",
                "total_pages": 1,
                "detection_summary": {
                    "table": 1,
                    "text_region": 3
                }
            },
            "detections": {
                "0": [
                    {
                        "class": "table",
                        "confidence": 0.85,
                        "bbox": [100, 100, 300, 300]
                    }
                ]
            }
        }
    
    def create_test_text_results(self):
        """Create test text results"""
        return {
            "success": True,
            "document_id": "test_doc_123",
            "text_summary": {
                "document_id": "test_doc_123",
                "ocr_statistics": {
                    "total_pages": 1,
                    "total_words": 50,
                    "average_confidence": 0.88
                }
            },
            "extracted_entities": {
                "dates": ["2024-01-15"],
                "amounts": ["$1,250.00"],
                "organizations": ["ABC Corporation"]
            }
        }
    
    @pytest.mark.asyncio
    async def test_document_processing(self):
        """Test complete fusion processing"""
        vision_results = self.create_test_vision_results()
        text_results = self.create_test_text_results()
        
        result = await self.fusion_agent.process_document(
            document_id="test_doc_123",
            vision_results=vision_results,
            text_results=text_results
        )
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "document_id" in result
        
        if result["success"]:
            assert "structured_output" in result
            assert "confidence_scores" in result
    
    def test_data_alignment_logic(self):
        """Test data alignment logic"""
        state = FusionState(
            document_id="test_doc",
            vision_results=self.create_test_vision_results(),
            text_results=self.create_test_text_results()
        )
        
        updated_state = self.fusion_agent.align_data(state)
        
        assert isinstance(updated_state, FusionState)
        assert "fused_results" in updated_state.dict()
        assert "aligned_data" in updated_state.fused_results
    
    def test_conflict_resolution_logic(self):
        """Test conflict resolution logic"""
        state = FusionState(
            document_id="test_doc",
            vision_results=self.create_test_vision_results(),
            text_results=self.create_test_text_results()
        )
        
        # First align data
        state = self.fusion_agent.align_data(state)
        
        # Then resolve conflicts
        updated_state = self.fusion_agent.resolve_conflicts(state)
        
        assert isinstance(updated_state, FusionState)
        assert "conflicts" in updated_state.dict()
    
    def test_confidence_computation_logic(self):
        """Test confidence computation logic"""
        state = FusionState(
            document_id="test_doc",
            vision_results=self.create_test_vision_results(),
            text_results=self.create_test_text_results()
        )
        
        # Run through pipeline
        state = self.fusion_agent.align_data(state)
        state = self.fusion_agent.resolve_conflicts(state)
        updated_state = self.fusion_agent.compute_confidence(state)
        
        assert isinstance(updated_state, FusionState)
        assert "confidence_scores" in updated_state.dict()
        assert len(updated_state.confidence_scores) > 0

class TestValidationAgent:
    """Test Validation Agent"""
    
    def setup_method(self):
        self.validation_agent = ValidationAgent()
    
    def create_test_fusion_results(self):
        """Create test fusion results"""
        return {
            "structured_output": {
                "document_id": "test_doc_123",
                "fields": {
                    "invoice_date": {
                        "value": "2024-01-15",
                        "confidence": 0.92,
                        "source": "OCR"
                    },
                    "invoice_amount": {
                        "value": "$1,250.00",
                        "confidence": 0.45,  # Low confidence
                        "source": "OCR"
                    },
                    "vendor_name": {
                        "value": "ABC Corporation",
                        "confidence": 0.88,
                        "source": "OCR"
                    }
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_document_validation(self):
        """Test complete document validation"""
        fusion_results = self.create_test_fusion_results()
        
        result = await self.validation_agent.validate_document(
            document_id="test_doc_123",
            fused_results=fusion_results
        )
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "document_id" in result
        
        if result["success"]:
            assert "validation_summary" in result
            assert "flags" in result
            assert "recommendations" in result
    
    def test_cross_check_logic(self):
        """Test cross-check logic"""
        state = ValidationState(
            document_id="test_doc",
            fused_results=self.create_test_fusion_results()
        )
        
        updated_state = self.validation_agent.cross_check(state)
        
        assert isinstance(updated_state, ValidationState)
        assert "validation_results" in updated_state.dict()
        assert "consistency_checks" in updated_state.validation_results
    
    def test_inconsistency_detection_logic(self):
        """Test inconsistency detection logic"""
        state = ValidationState(
            document_id="test_doc",
            fused_results=self.create_test_fusion_results()
        )
        
        # First cross-check
        state = self.validation_agent.cross_check(state)
        
        # Then detect inconsistencies
        updated_state = self.validation_agent.detect_inconsistencies(state)
        
        assert isinstance(updated_state, ValidationState)
        assert "inconsistencies" in updated_state.validation_results
    
    def test_flag_generation_logic(self):
        """Test flag generation logic"""
        state = ValidationState(
            document_id="test_doc",
            fused_results=self.create_test_fusion_results()
        )
        
        # Run through pipeline
        state = self.validation_agent.cross_check(state)
        state = self.validation_agent.detect_inconsistencies(state)
        updated_state = self.validation_agent.generate_flags(state)
        
        assert isinstance(updated_state, ValidationState)
        assert "flags" in updated_state.dict()
        assert len(updated_state.flags) > 0
        
        # Check flag structure
        for flag in updated_state.flags:
            assert "type" in flag
            assert "field" in flag
            assert "status" in flag
            assert "reason" in flag
            assert "priority" in flag
    
    def test_recommendation_logic(self):
        """Test recommendation logic"""
        state = ValidationState(
            document_id="test_doc",
            fused_results=self.create_test_fusion_results()
        )
        
        # Run through pipeline
        state = self.validation_agent.cross_check(state)
        state = self.validation_agent.detect_inconsistencies(state)
        state = self.validation_agent.generate_flags(state)
        updated_state = self.validation_agent.provide_recommendations(state)
        
        assert isinstance(updated_state, ValidationState)
        assert "recommendations" in updated_state.dict()
        assert len(updated_state.recommendations) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])