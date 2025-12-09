from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import tempfile
import os

class PDFGenerator:
    def create_report(self, analysis_data: dict) -> str:
        """
        Create PDF report from analysis data
        """
        # Create temporary file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.pdf')
        os.close(tmp_fd)

        try:
            # Create PDF
            doc = SimpleDocTemplate(tmp_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            story.append(Paragraph("SpeakPace Analysis Report", styles['Title']))
            story.append(Spacer(1, 12))

            # Summary
            story.append(Paragraph(f"Duration: {analysis_data['duration']:.2f} seconds", styles['Normal']))
            story.append(Paragraph(f"Word Count: {analysis_data['wordCount']}", styles['Normal']))
            story.append(Paragraph(f"Average Pace: {analysis_data['avgPace']} WPM", styles['Normal']))
            story.append(Paragraph(f"Total Fillers: {analysis_data['totalFillers']}", styles['Normal']))
            story.append(Spacer(1, 12))

            # Pace Data
            if analysis_data.get('paceData'):
                story.append(Paragraph("Pace Analysis:", styles['Heading2']))
                for point in analysis_data['paceData'][:5]:  # Show first 5 points
                    story.append(Paragraph(f"Time: {point['time']}s - {point['words_per_minute']} WPM", styles['Normal']))
                story.append(Spacer(1, 12))

            # Filler Words
            if analysis_data.get('fillerWords'):
                story.append(Paragraph("Filler Words:", styles['Heading2']))
                for filler in analysis_data['fillerWords']:
                    story.append(Paragraph(f"{filler['word']}: {filler['count']}", styles['Normal']))
                story.append(Spacer(1, 12))

            # Emphasis Data
            if analysis_data.get('emphasisData'):
                story.append(Paragraph("Emphasis Points:", styles['Heading2']))
                for emphasis in analysis_data['emphasisData'][:5]:  # Show first 5 points
                    story.append(Paragraph(f"Time: {emphasis['time']:.1f}s - Intensity: {emphasis['intensity']:.2f}", styles['Normal']))
                story.append(Spacer(1, 12))

            # Transcript
            story.append(Paragraph("Transcript:", styles['Heading2']))
            transcript_text = analysis_data.get('transcript', 'No transcript available')
            story.append(Paragraph(transcript_text, styles['Normal']))

            doc.build(story)
            return tmp_path

        except Exception as e:
            # Clean up on error
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise e
