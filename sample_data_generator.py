import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random

# Try to import reportlab, create fallbacks if not available
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("Warning: reportlab not installed, creating dummy functions")

class SampleDataGenerator:
    """Generate synthetic test documents with contradictions"""
    
    def __init__(self, output_dir="sample_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_contradictory_document(self, filename="synthetic_test_document.pdf"):
        """
        Generate a PDF with chart and contradictory text
        
        Purpose: To demonstrate Validation Agent detecting contradictions
        """
        if not HAS_REPORTLAB:
            print("Error: reportlab required for PDF generation")
            return None
            
        # Create PDF
        pdf_path = os.path.join(self.output_dir, filename)
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        
        # Add title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, height - 50, "Quarterly Financial Report - Q4 2024")
        
        # Add contradictory text paragraph
        c.setFont("Helvetica", 12)
        text = (
            "The company experienced a significant decline in revenue during Q4 2024, "
            "with total revenue decreasing by 15% compared to Q3. This decline was "
            "primarily due to reduced customer demand and increased competition in "
            "the market. The downward trend is expected to continue into Q1 2025."
        )
        
        # Split text into lines
        text_lines = []
        words = text.split()
        current_line = []
        current_width = 0
        
        for word in words:
            word_width = c.stringWidth(word + " ", "Helvetica", 12)
            if current_width + word_width > 400:
                text_lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width
            else:
                current_line.append(word)
                current_width += word_width
        
        if current_line:
            text_lines.append(" ".join(current_line))
        
        # Draw text
        y_position = height - 100
        for line in text_lines:
            c.drawString(100, y_position, line)
            y_position -= 20
        
        # Generate contradictory chart image
        chart_image = self._generate_contradictory_chart()
        
        # Save chart to temporary file
        chart_path = os.path.join(self.output_dir, "temp_chart.png")
        chart_image.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close(chart_image)
        
        # Add chart to PDF
        img = ImageReader(chart_path)
        c.drawImage(img, 100, height - 400, width=400, height=250)
        
        # Add chart caption that matches the chart (contradicting the text)
        c.setFont("Helvetica", 10)
        c.drawString(100, height - 420, "Figure 1: Quarterly Revenue Trend (Showing Growth)")
        
        # Add a table with conflicting data
        self._add_contradictory_table(c, 100, height - 500)
        
        # Add metadata section
        c.setFont("Helvetica-Bold", 12)
        c.drawString(100, height - 550, "Document Metadata:")
        c.setFont("Helvetica", 10)
        
        metadata = [
            ("Generated Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("Document Type:", "Synthetic Test Document"),
            ("Purpose:", "Validation Agent Testing"),
            ("Contradiction Type:", "Text claims decline, chart shows growth")
        ]
        
        y_pos = height - 570
        for label, value in metadata:
            c.drawString(100, y_pos, f"{label} {value}")
            y_pos -= 15
        
        # Add validation instructions
        c.setFont("Helvetica-Bold", 12)
        c.drawString(100, height - 630, "Expected Validation Results:")
        c.setFont("Helvetica", 10)
        
        instructions = [
            "1. Vision Agent should detect: chart, table, text regions",
            "2. Text Agent should extract: dates, amounts, trend descriptions",
            "3. Fusion Agent should identify: revenue trend data",
            "4. Validation Agent should flag: CONTRADICTION between text and chart",
            "5. Confidence for revenue_trend field should be: LOW_CONFIDENCE"
        ]
        
        y_pos = height - 650
        for instruction in instructions:
            c.drawString(110, y_pos, instruction)
            y_pos -= 15
        
        # Save PDF
        c.save()
        
        # Clean up temporary chart file
        if os.path.exists(chart_path):
            os.remove(chart_path)
        
        print(f"Generated contradictory document: {pdf_path}")
        return pdf_path
    
    def _generate_contradictory_chart(self):
        """Generate a chart showing growth (contradicting the text)"""
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Quarterly data (showing growth, contradicting the text)
        quarters = ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']
        revenue = [45.2, 48.7, 52.3, 60.1]  # Increasing revenue (in millions)
        
        # Create bar chart
        bars = ax.bar(quarters, revenue, color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'])
        
        # Add value labels on bars
        for bar, value in zip(bars, revenue):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'${value}M', ha='center', va='bottom', fontweight='bold')
        
        # Customize chart
        ax.set_ylabel('Revenue (Millions USD)', fontweight='bold')
        ax.set_xlabel('Quarter', fontweight='bold')
        ax.set_title('Quarterly Revenue Trend - Steady Growth', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add growth percentage annotations
        for i in range(1, len(revenue)):
            growth = ((revenue[i] - revenue[i-1]) / revenue[i-1]) * 100
            ax.annotate(f'+{growth:.1f}%', 
                       xy=(i, revenue[i]), 
                       xytext=(i, revenue[i] + 2),
                       ha='center',
                       fontweight='bold',
                       color='green' if growth > 0 else 'red')
        
        plt.tight_layout()
        return fig
    
    def _add_contradictory_table(self, canvas, x, y):
        """Add a table with data that contradicts the text"""
        # Table data (showing growth)
        table_data = [
            ["Quarter", "Revenue ($M)", "Growth %", "Comments"],
            ["Q1 2024", "45.2", "-", "Baseline"],
            ["Q2 2024", "48.7", "+7.7%", "Steady growth"],
            ["Q3 2024", "52.3", "+7.4%", "Accelerating"],
            ["Q4 2024", "60.1", "+14.9%", "Strong finish"]
        ]
        
        # Draw table
        canvas.setFont("Helvetica-Bold", 12)
        canvas.drawString(x, y, "Quarterly Performance Summary:")
        
        y -= 20
        canvas.setFont("Helvetica", 10)
        
        # Draw table headers
        headers = table_data[0]
        col_widths = [120, 100, 80, 150]
        
        current_x = x
        for i, header in enumerate(headers):
            canvas.setFont("Helvetica-Bold", 10)
            canvas.drawString(current_x, y, header)
            current_x += col_widths[i]
        
        # Draw table rows
        y -= 20
        for row in table_data[1:]:
            current_x = x
            canvas.setFont("Helvetica", 10)
            
            for i, cell in enumerate(row):
                # Color code growth percentages
                if i == 2 and cell != "-":
                    if "+" in cell:
                        canvas.setFillColorRGB(0, 0.5, 0)  # Green for positive
                    else:
                        canvas.setFillColorRGB(0.8, 0, 0)  # Red for negative
                
                canvas.drawString(current_x, y, cell)
                canvas.setFillColorRGB(0, 0, 0)  # Reset to black
                current_x += col_widths[i]
            
            y -= 15
    
    def generate_multiple_test_documents(self, count=5):
        """Generate multiple test documents with varying contradictions"""
        documents = []
        
        contradiction_types = [
            ("revenue_trend", "text claims decline, chart shows growth"),
            ("expense_report", "table total doesn't match sum of line items"),
            ("signature_verification", "signature differs from reference"),
            ("date_inconsistency", "dates in header and footer don't match"),
            ("amount_discrepancy", "numeric amount in text vs table differs")
        ]
        
        for i in range(min(count, len(contradiction_types))):
            doc_type, description = contradiction_types[i]
            filename = f"test_doc_{i+1}_{doc_type}.pdf"
            
            # Customize document based on contradiction type
            if doc_type == "revenue_trend":
                doc_path = self.generate_contradictory_document(filename)
            else:
                doc_path = self._generate_generic_document(filename, doc_type, description)
            
            if doc_path:
                documents.append({
                    "path": doc_path,
                    "type": doc_type,
                    "description": description,
                    "expected_validation_flags": [doc_type.upper()]
                })
        
        # Create manifest file
        self._create_manifest(documents)
        
        return documents
    
    def _generate_generic_document(self, filename, doc_type, description):
        """Generate a generic test document"""
        if not HAS_REPORTLAB:
            return None
            
        pdf_path = os.path.join(self.output_dir, filename)
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        
        # Add title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, height - 50, f"Test Document: {doc_type.replace('_', ' ').title()}")
        
        # Add description
        c.setFont("Helvetica", 12)
        c.drawString(100, height - 80, f"Contradiction: {description}")
        
        # Add sample content based on type
        if doc_type == "expense_report":
            self._add_expense_report(c, 100, height - 120)
        elif doc_type == "signature_verification":
            self._add_signature_section(c, 100, height - 120)
        elif doc_type == "date_inconsistency":
            self._add_date_inconsistencies(c, 100, height - 120)
        elif doc_type == "amount_discrepancy":
            self._add_amount_discrepancy(c, 100, height - 120)
        
        # Add validation instructions
        c.setFont("Helvetica-Bold", 12)
        c.drawString(100, height - 300, "Expected Validation Flags:")
        c.setFont("Helvetica", 10)
        c.drawString(100, height - 320, f"- {doc_type.upper()}_FLAG")
        c.drawString(100, height - 340, "- INCONSISTENCY_DETECTED")
        
        c.save()
        return pdf_path
    
    def _add_expense_report(self, canvas, x, y):
        """Add an expense report with calculation errors"""
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawString(x, y, "Expense Report")
        
        y -= 30
        canvas.setFont("Helvetica", 10)
        
        # Table with calculation error
        expenses = [
            ["Item", "Amount"],
            ["Travel", "$1,250.00"],
            ["Meals", "$845.50"],
            ["Supplies", "$320.75"],
            ["Software", "$1,500.00"],
            ["Total", "$3,916.25"]  # Wrong total (should be $3,916.25)
        ]
        
        for row in expenses:
            canvas.drawString(x, y, row[0])
            canvas.drawString(x + 200, y, row[1])
            y -= 20
        
        # Note about the error
        y -= 10
        canvas.setFont("Helvetica-Oblique", 9)
        canvas.drawString(x, y, "Note: The total should be $3,916.25 but is listed as $3,916.25")
    
    def _add_signature_section(self, canvas, x, y):
        """Add signature section with verification note"""
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawString(x, y, "Authorization Section")
        
        y -= 40
        canvas.setFont("Helvetica", 10)
        canvas.drawString(x, y, "Authorized Signature:")
        
        # Draw signature line
        canvas.line(x, y - 5, x + 200, y - 5)
        
        y -= 30
        canvas.drawString(x, y, "Reference Signature (from database):")
        canvas.line(x, y - 5, x + 200, y - 5)
        
        y -= 20
        canvas.setFont("Helvetica-Oblique", 9)
        canvas.drawString(x, y, "Note: Signatures do not match reference records")
    
    def _add_date_inconsistencies(self, canvas, x, y):
        """Add document with date inconsistencies"""
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawString(x, y, "Document with Date Fields")
        
        y -= 40
        canvas.setFont("Helvetica", 10)
        
        dates = [
            ("Document Date (Header):", "2024-01-15"),
            ("Effective Date:", "2024-01-10"),
            ("Approval Date (Footer):", "2024-01-20"),
            ("Invoice Date:", "2024-01-12")
        ]
        
        for label, date in dates:
            canvas.drawString(x, y, label)
            canvas.drawString(x + 200, y, date)
            y -= 20
        
        y -= 10
        canvas.setFont("Helvetica-Oblique", 9)
        canvas.drawString(x, y, "Note: Dates are inconsistent across document")
    
    def _add_amount_discrepancy(self, canvas, x, y):
        """Add document with amount discrepancies"""
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawString(x, y, "Financial Summary")
        
        y -= 40
        canvas.setFont("Helvetica", 10)
        
        amounts = [
            ("Total in text paragraph:", "$12,500.00"),
            ("Total in table:", "$12,750.00"),
            ("Total in summary box:", "$12,450.00")
        ]
        
        for label, amount in amounts:
            canvas.drawString(x, y, label)
            canvas.drawString(x + 200, y, amount)
            y -= 20
        
        y -= 10
        canvas.setFont("Helvetica-Oblique", 9)
        canvas.drawString(x, y, "Note: Amounts differ across document sections")
    
    def _create_manifest(self, documents):
        """Create manifest file describing test documents"""
        manifest_path = os.path.join(self.output_dir, "test_documents_manifest.json")
        
        import json
        manifest = {
            "generated_date": datetime.now().isoformat(),
            "total_documents": len(documents),
            "documents": documents,
            "validation_expectations": {
                "vision_agent": "Detect tables, charts, signatures, text regions",
                "text_agent": "Extract entities with confidence scores",
                "fusion_agent": "Align modalities and compute confidence",
                "validation_agent": "Flag inconsistencies and contradictions"
            }
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Created manifest: {manifest_path}")

if __name__ == "__main__":
    # Generate sample documents
    generator = SampleDataGenerator()
    
    print("Generating synthetic test documents...")
    
    # Generate the main contradictory document
    main_doc = generator.generate_contradictory_document()
    
    # Generate additional test documents
    additional_docs = generator.generate_multiple_test_documents(3)
    
    print(f"\nGenerated {len(additional_docs) + 1 if main_doc else 0} test documents:")
    if main_doc:
        print(f"1. {os.path.basename(main_doc)} - Main contradiction demo")
    for i, doc in enumerate(additional_docs, 2 if main_doc else 1):
        print(f"{i}. {os.path.basename(doc['path'])} - {doc['description']}")
    
    print(f"\nAll documents saved to: {generator.output_dir}")
    print("\nUse these documents to test the Validation Agent's contradiction detection.")