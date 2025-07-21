from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, after_this_request, Response
import os
import subprocess
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import io
import uuid
from werkzeug.utils import secure_filename
# Try to import rembg for background removal
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("Warning: rembg library not available. Background removal will not work.")

# Try to import additional AI models for enhanced processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Some enhancements will be disabled.")
import tempfile
import PyPDF2
from pdf2docx import Converter
from docx import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
import zipfile
import yt_dlp
import time
import shutil
import numpy as np

# Try to import PyMuPDF for better compression
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_file('static/favicon.svg', mimetype='image/svg+xml')

@app.route('/compress')
def compress():
    return render_template('compress.html')

@app.route('/resize')
def resize():
    return render_template('resize.html')

@app.route('/convert')
def convert():
    return render_template('convert.html')

@app.route('/crop')
def crop():
    return render_template('crop.html')

@app.route('/rotate')
def rotate():
    return render_template('rotate.html')

@app.route('/watermark')
def watermark():
    return render_template('watermark.html')

@app.route('/background-remover')
def background_remover():
    return render_template('background-remover.html')

@app.route('/edit-pdf')
def edit_pdf():
    return render_template('edit-pdf.html')

@app.route('/pdf-to-word')
def pdf_to_word():
    return render_template('pdf-to-word.html')

@app.route('/word-to-pdf')
def word_to_pdf():
    return render_template('word-to-pdf.html')

@app.route('/images-to-pdf')
def images_to_pdf():
    return render_template('images-to-pdf.html')

@app.route('/merge-pdf')
def merge_pdf():
    return render_template('merge-pdf.html')

@app.route('/compress-pdf')
def compress_pdf():
    return render_template('compress-pdf.html')

@app.route('/protect-pdf')
def protect_pdf():
    return render_template('protect-pdf.html')

@app.route('/remove-pdf-password')
def remove_pdf_password():
    return render_template('remove-pdf-password.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/video-downloader')
def video_downloader():
    return render_template('video-downloader.html')

@app.route('/ai-tools')
def ai_tools():
    return render_template('ai-tools.html')

@app.route('/ai-tools/<category>')
def ai_tools_category(category):
    return render_template('ai-tools.html', category=category)

# API endpoints for image processing
@app.route('/api/compress', methods=['POST'])
def api_compress():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    quality = request.form.get('quality', 80, type=int)
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported'}), 400
    
    try:
        # Read image
        img = Image.open(file.stream)
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        
        # Create output buffer
        output = io.BytesIO()
        
        # Save with compression using the quality parameter
        img.save(output, format='JPEG', quality=quality, optimize=True)
        output.seek(0)
        
        # Generate unique filename
        filename = f"compressed_{secure_filename(file.filename)}"
        if not filename.lower().endswith('.jpg'):
            filename = filename.rsplit('.', 1)[0] + '.jpg'
        
        return send_file(
            output,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/resize', methods=['POST'])
def api_resize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    width = request.form.get('width', type=int)
    height = request.form.get('height', type=int)
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported'}), 400
    
    try:
        # Read image
        img = Image.open(file.stream)
        
        # Calculate new size
        if width and height:
            new_size = (width, height)
        elif width:
            ratio = width / img.width
            new_size = (width, int(img.height * ratio))
        elif height:
            ratio = height / img.height
            new_size = (int(img.width * ratio), height)
        else:
            return jsonify({'error': 'Please specify width or height'}), 400
        
        # Resize image
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create output buffer
        output = io.BytesIO()
        
        # Save image
        img.save(output, format='PNG')
        output.seek(0)
        
        # Generate filename
        filename = f"resized_{secure_filename(file.filename)}"
        if not filename.lower().endswith('.png'):
            filename = filename.rsplit('.', 1)[0] + '.png'
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/convert', methods=['POST'])
def api_convert():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    target_format = request.form.get('format', 'png').lower()
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported'}), 400
    
    try:
        # Read image
        img = Image.open(file.stream)
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P') and target_format in ('jpg', 'jpeg'):
            img = img.convert('RGB')
        
        # Create output buffer
        output = io.BytesIO()
        
        # Save in target format
        if target_format == 'jpg':
            img.save(output, format='JPEG', quality=95)
            mimetype = 'image/jpeg'
            ext = '.jpg'
        elif target_format == 'png':
            img.save(output, format='PNG')
            mimetype = 'image/png'
            ext = '.png'
        elif target_format == 'webp':
            img.save(output, format='WebP')
            mimetype = 'image/webp'
            ext = '.webp'
        elif target_format == 'gif':
            img.save(output, format='GIF')
            mimetype = 'image/gif'
            ext = '.gif'
        else:
            return jsonify({'error': 'Unsupported format'}), 400
        
        output.seek(0)
        
        # Generate filename
        filename = f"converted_{secure_filename(file.filename)}"
        filename = filename.rsplit('.', 1)[0] + ext
        
        return send_file(
            output,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crop', methods=['POST'])
def api_crop():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    width = request.form.get('width', type=int)
    height = request.form.get('height', type=int)
    
    # Get crop coordinates from live crop
    crop_x = request.form.get('crop_x', type=int)
    crop_y = request.form.get('crop_y', type=int)
    crop_width = request.form.get('crop_width', type=int)
    crop_height = request.form.get('crop_height', type=int)
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported'}), 400
    
    try:
        # Read image
        img = Image.open(file.stream)
        
        # Calculate crop dimensions
        img_width, img_height = img.size
        
        # Check if we have crop coordinates from live crop
        if crop_x is not None and crop_y is not None and crop_width is not None and crop_height is not None:
            # Use the exact crop coordinates provided
            left = crop_x
            top = crop_y
            right = crop_x + crop_width
            bottom = crop_y + crop_height
            
            print(f"Crop coordinates: x={crop_x}, y={crop_y}, w={crop_width}, h={crop_height}")
            print(f"Crop box: left={left}, top={top}, right={right}, bottom={bottom}")
            
        elif width and height:
            # Center crop to specified dimensions
            left = (img_width - width) // 2
            top = (img_height - height) // 2
            right = left + width
            bottom = top + height
        elif width:
            # Crop to specified width, maintain aspect ratio
            ratio = width / img_width
            height = int(img_height * ratio)
            left = 0
            top = 0
            right = width
            bottom = height
        elif height:
            # Crop to specified height, maintain aspect ratio
            ratio = height / img_height
            width = int(img_width * ratio)
            left = 0
            top = 0
            right = width
            bottom = height
        else:
            return jsonify({'error': 'Please specify width or height, or use live crop preview'}), 400
        
        # Ensure crop box is within image bounds
        left = max(0, left)
        top = max(0, top)
        right = min(img_width, right)
        bottom = min(img_height, bottom)
        
        # Crop image
        img = img.crop((left, top, right, bottom))
        
        # Create output buffer
        output = io.BytesIO()
        
        # Save image
        img.save(output, format='PNG')
        output.seek(0)
        
        # Generate filename
        filename = f"cropped_{secure_filename(file.filename)}"
        if not filename.lower().endswith('.png'):
            filename = filename.rsplit('.', 1)[0] + '.png'
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/rotate', methods=['POST'])
def api_rotate():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    angle = request.form.get('angle', type=float)
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported'}), 400
    
    try:
        # Read image
        img = Image.open(file.stream)
        
        # Rotate image
        img = img.rotate(-angle, expand=True)  # Negative angle for clockwise rotation
        
        # Create output buffer
        output = io.BytesIO()
        
        # Save image
        img.save(output, format='PNG')
        output.seek(0)
        
        # Generate filename
        filename = f"rotated_{secure_filename(file.filename)}"
        if not filename.lower().endswith('.png'):
            filename = filename.rsplit('.', 1)[0] + '.png'
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/watermark', methods=['POST'])
def api_watermark():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    watermark_text = request.form.get('text', '')
    position = request.form.get('position', 'bottom-right')
    opacity = request.form.get('opacity', '80', type=int)
    color = request.form.get('color', '#ffffff')
    fontSize = request.form.get('fontSize', '36', type=int)
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported'}), 400
    
    if not watermark_text:
        return jsonify({'error': 'Watermark text is required'}), 400
    
    try:
        # Read image
        img = Image.open(file.stream)
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        
        # Create drawing object
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", fontSize)
        except:
            font = ImageFont.load_default()
        
        # Get text size
        bbox = draw.textbbox((0, 0), watermark_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate position
        img_width, img_height = img.size
        margin = 20
        
        if position == 'top-left':
            x, y = margin, margin
        elif position == 'top-center':
            x, y = (img_width - text_width) // 2, margin
        elif position == 'top-right':
            x, y = img_width - text_width - margin, margin
        elif position == 'center-left':
            x, y = margin, (img_height - text_height) // 2
        elif position == 'center':
            x, y = (img_width - text_width) // 2, (img_height - text_height) // 2
        elif position == 'center-right':
            x, y = img_width - text_width - margin, (img_height - text_height) // 2
        elif position == 'bottom-left':
            x, y = margin, img_height - text_height - margin
        elif position == 'bottom-center':
            x, y = (img_width - text_width) // 2, img_height - text_height - margin
        else:  # bottom-right
            x, y = img_width - text_width - margin, img_height - text_height - margin
        
        # Convert hex color to RGB
        color = color.lstrip('#')
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        
        # Calculate alpha based on opacity
        alpha = int((opacity / 100) * 255)
        
        # Draw watermark with custom color and opacity
        draw.text((x, y), watermark_text, fill=(r, g, b, alpha), font=font)
        
        # Create output buffer
        output = io.BytesIO()
        
        # Save image
        img.save(output, format='PNG')
        output.seek(0)
        
        # Generate filename
        filename = f"watermarked_{secure_filename(file.filename)}"
        if not filename.lower().endswith('.png'):
            filename = filename.rsplit('.', 1)[0] + '.png'
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/remove-background', methods=['POST'])
def api_remove_background():
    if not REMBG_AVAILABLE:
        return jsonify({'error': 'Background removal service is not available. Please contact support.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported'}), 400
    
    try:
        # Read image
        input_image = Image.open(file.stream)
        
        # Convert to RGB if necessary
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        # Store original size for final output
        original_size = input_image.size
        
        # Optimized resolution for fast processing while maintaining quality
        max_size = 2048  # Reduced for faster processing
        if max(input_image.size) > max_size:
            ratio = max_size / max(input_image.size)
            new_size = (int(input_image.width * ratio), int(input_image.height * ratio))
            input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
        
        print("Starting remove.bg-quality background removal...")
        
        # Fast but high-quality background removal
        print("Starting fast background removal...")
        output_image = remove(
            input_image,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=5  # Reduced for speed
        )
        
        # Convert to RGBA
        if output_image.mode != 'RGBA':
            output_image = output_image.convert('RGBA')
        
        # Convert to numpy for advanced processing
        img_array = np.array(output_image)
        
        # Advanced artifact removal and edge refinement
        print("Applying remove.bg-quality edge refinement...")
        
        # Get alpha channel
        alpha = img_array[:, :, 3]
        
        # Remove any color bleeding artifacts
        print("Removing color bleeding artifacts...")
        
        # Create masks for different color artifacts
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        
        # Green artifact removal (more aggressive)
        green_mask = (g > r * 1.1) & (g > b * 1.1) & (alpha < 200)
        img_array[green_mask, 3] = 0
        
        # Red artifact removal
        red_mask = (r > g * 1.1) & (r > b * 1.1) & (alpha < 200)
        img_array[red_mask, 3] = 0
        
        # Blue artifact removal
        blue_mask = (b > r * 1.1) & (b > g * 1.1) & (alpha < 200)
        img_array[blue_mask, 3] = 0
        
        # Fast edge refinement using OpenCV if available
        if CV2_AVAILABLE:
            print("Applying fast edge refinement...")
            
            # Convert alpha to OpenCV format
            alpha_cv = alpha.astype(np.uint8)
            
            # Fast morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # Smaller kernel for speed
            
            # Quick hole filling
            alpha_cv = cv2.morphologyEx(alpha_cv, cv2.MORPH_CLOSE, kernel)
            
            # Fast bilateral filter
            alpha_cv = cv2.bilateralFilter(alpha_cv, 5, 50, 50)  # Reduced parameters for speed
            
            # Quick edge enhancement
            gaussian = cv2.GaussianBlur(alpha_cv, (0, 0), 1.5)  # Reduced blur for speed
            alpha_cv = cv2.addWeighted(alpha_cv, 1.3, gaussian, -0.3, 0)  # Reduced enhancement for speed
            
            # Ensure values are in valid range
            alpha_cv = np.clip(alpha_cv, 0, 255)
            
            # Update alpha channel
            img_array[:, :, 3] = alpha_cv
        
        # Convert back to PIL
        final_image = Image.fromarray(img_array.astype(np.uint8))
        
        # Quick final enhancement
        print("Applying quick final enhancement...")
        
        # Get the alpha channel
        final_alpha = final_image.split()[-1]
        
        # Quick contrast enhancement
        enhancer = ImageEnhance.Contrast(final_alpha)
        enhanced_alpha = enhancer.enhance(1.2)  # Reduced enhancement for speed
        
        # Create final image
        r, g, b, _ = final_image.split()
        final_image = Image.merge('RGBA', (r, g, b, enhanced_alpha))
        
        # Resize back to original size with high-quality resampling
        if final_image.size != original_size:
            final_image = final_image.resize(original_size, Image.Resampling.LANCZOS)
        
        # Create output buffer
        output = io.BytesIO()
        
        # Save with optimized quality for speed
        print("Saving with optimized settings...")
        final_image.save(output, format='PNG', optimize=True, compress_level=6)
        output.seek(0)
        
        # Generate filename
        filename = f"fast_no_bg_{secure_filename(file.filename)}"
        if not filename.lower().endswith('.png'):
            filename = filename.rsplit('.', 1)[0] + '.png'
        
        print("Fast background removal completed!")
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        print(f"Fast background removal failed: {e}")
        print("Falling back to basic background removal...")
        
        try:
            # Fallback to enhanced background removal
            file.stream.seek(0)
            input_image = Image.open(file.stream)
            
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
            
            # Basic background removal
            output_image = remove(input_image)
            
            # Create output buffer
            output = io.BytesIO()
            output_image.save(output, format='PNG', optimize=False)
            output.seek(0)
            
            # Generate filename
            filename = f"basic_no_bg_{secure_filename(file.filename)}"
            if not filename.lower().endswith('.png'):
                filename = filename.rsplit('.', 1)[0] + '.png'
            
            print("Basic background removal completed successfully!")
            return send_file(
                output,
                mimetype='image/png',
                as_attachment=True,
                download_name=filename
            )
            
        except Exception as fallback_error:
            print(f"Basic background removal also failed: {fallback_error}")
            return jsonify({'error': 'Background removal failed. Please try with a different image.'}), 500

@app.route('/api/background-remover', methods=['POST'])
def background_remover_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read the image
        input_image = Image.open(file.stream)
        
        # Remove background
        output_image = remove(input_image)
        
        # Convert to RGB if necessary
        if output_image.mode != 'RGB':
            output_image = output_image.convert('RGB')
        
        # Save to bytes
        img_io = io.BytesIO()
        output_image.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(
            img_io,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'removed_bg_{secure_filename(file.filename)}'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# PDF Tools API Endpoints

@app.route('/api/pdf-to-word', methods=['POST'])
def pdf_to_word_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file is a PDF
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Please upload a PDF file'}), 400
    
    pdf_path = None
    docx_path = None
    cv = None
    
    try:
        # Save uploaded PDF to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
            file.save(tmp_pdf.name)
            pdf_path = tmp_pdf.name
        
        # Create temporary Word file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_docx:
            docx_path = tmp_docx.name
        
        # Convert PDF to Word using pdf2docx
        cv = Converter(pdf_path)
        cv.convert(docx_path, start=0, end=None)
        cv.close()
        
        # Check if conversion was successful
        if os.path.exists(docx_path) and os.path.getsize(docx_path) > 0:
            # Return the Word document
            return send_file(
                docx_path,
                mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                as_attachment=True,
                download_name=f'{os.path.splitext(secure_filename(file.filename))[0]}.docx'
            )
        else:
            return jsonify({'error': 'Conversion failed - no output file generated'}), 500
    
    except Exception as e:
        print(f"PDF to Word conversion error: {e}")
        return jsonify({'error': f'Conversion failed: {str(e)}'}), 500
    finally:
        # Clean up temporary files
        if cv:
            try:
                cv.close()
            except:
                pass
        if pdf_path:
            try:
                os.unlink(pdf_path)
            except:
                pass

@app.route('/api/word-to-pdf', methods=['POST'])
def word_to_pdf_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file is a Word document
    if not file.filename.lower().endswith(('.docx', '.doc')):
        return jsonify({'error': 'Please upload a Word document (.docx or .doc)'}), 400
    
    doc = None
    pdf_path = None
    
    try:
        # Load the Word document
        doc = Document(file)
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
            pdf_path = tmp_pdf.name
        
        # Create PDF with better formatting
        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4
        
        # Set font and size
        c.setFont("Helvetica", 12)
        
        y_position = height - 50
        line_height = 15
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                # Handle text wrapping
                text = paragraph.text
                words = text.split()
                lines = []
                current_line = ""
                
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    if c.stringWidth(test_line, "Helvetica", 12) < width - 100:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
                
                # Draw each line
                for line in lines:
                    if y_position < 50:  # New page
                        c.showPage()
                        c.setFont("Helvetica", 12)
                        y_position = height - 50
                    
                    c.drawString(50, y_position, line)
                    y_position -= line_height
                
                # Add extra space between paragraphs
                y_position -= 5
        
        c.save()
        
        # Check if PDF was created successfully
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            # Try to improve the PDF using PyMuPDF if available
            if PYMUPDF_AVAILABLE:
                try:
                    # Open the generated PDF
                    doc = fitz.open(pdf_path)
                    
                    # Create a new PDF with better formatting
                    new_doc = fitz.open()
                    
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
                        new_page.show_pdf_page(new_page.rect, doc, page_num)
                    
                    # Save with better compression
                    new_doc.save(pdf_path + '_improved',
                               garbage=4,
                               deflate=True,
                               clean=True,
                               linear=True)
                    
                    new_doc.close()
                    doc.close()
                    
                    # Use improved version if it's better
                    if os.path.exists(pdf_path + '_improved'):
                        try:
                            os.unlink(pdf_path)
                            os.rename(pdf_path + '_improved', pdf_path)
                        except:
                            pass
                except Exception as e:
                    print(f"PDF improvement failed: {e}")
                    pass
            
            # Return the PDF
            return send_file(
                pdf_path,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'{os.path.splitext(secure_filename(file.filename))[0]}.pdf'
            )
        else:
            return jsonify({'error': 'PDF creation failed - no output file generated'}), 500
    
    except Exception as e:
        print(f"Word to PDF conversion error: {e}")
        return jsonify({'error': f'Conversion failed: {str(e)}'}), 500
    finally:
        # Clean up
        if doc:
            try:
                del doc
            except:
                pass

@app.route('/api/images-to-pdf', methods=['POST'])
def images_to_pdf_api():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    temp_files = []
    try:
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
            pdf_path = tmp_pdf.name
        
        # Create PDF
        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4
        
        for file in files:
            if file.filename:
                # Open image
                img = Image.open(file)
                
                # Get image dimensions
                img_width, img_height = img.size
                aspect = img_width / img_height
                
                # Define page margins (in points)
                margin = 50
                available_width = width - (2 * margin)
                available_height = height - (2 * margin)
                
                # Calculate scaling to fit image within available space
                scale_width = available_width / img_width
                scale_height = available_height / img_height
                scale = min(scale_width, scale_height)  # Use the smaller scale to fit completely
                
                # Calculate new dimensions
                new_width = img_width * scale
                new_height = img_height * scale
                
                # Center image on page
                x = margin + (available_width - new_width) / 2
                y = margin + (available_height - new_height) / 2
                
                # Save image to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
                    img.save(tmp_img.name, 'PNG')
                    temp_files.append(tmp_img.name)
                    c.drawImage(tmp_img.name, x, y, new_width, new_height)
                
                c.showPage()
        
        c.save()
        
        # Clean up temporary image files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        # Return the PDF
        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='images.pdf'
        )
    
    except Exception as e:
        # Clean up temporary files on error
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        return jsonify({'error': str(e)}), 500

@app.route('/api/merge-pdf', methods=['POST'])
def merge_pdf_api():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    if len(files) < 2:
        return jsonify({'error': 'At least 2 PDF files required'}), 400
    
    try:
        # Create PDF merger
        merger = PyPDF2.PdfMerger()
        
        # Add each PDF to merger
        for file in files:
            if file.filename:
                merger.append(file)
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
            output_path = tmp_pdf.name
        
        # Write merged PDF
        with open(output_path, 'wb') as output_file:
            merger.write(output_file)
        
        merger.close()
        
        # Return the merged PDF
        return send_file(
            output_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='merged.pdf'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compress-pdf', methods=['POST'])
def compress_pdf_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get quality parameter (10-100)
    quality = int(request.form.get('quality', 70))
    quality = max(10, min(100, quality))  # Clamp between 10 and 100
    
    # Store original filename
    original_filename = secure_filename(file.filename)
    
    try:
        # Save uploaded PDF to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
            file.save(tmp_pdf.name)
            pdf_path = tmp_pdf.name
        
        # Get original file size
        original_size = os.path.getsize(pdf_path)
        print(f"Original file size: {original_size} bytes")
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as compressed_pdf:
            output_path = compressed_pdf.name
        
        # Calculate compression parameters based on quality
        if quality >= 80:
            image_resolution = 150
            image_quality = 80
            pdf_settings = '/ebook'
        elif quality >= 60:
            image_resolution = 100
            image_quality = 60
            pdf_settings = '/screen'
        elif quality >= 40:
            image_resolution = 72
            image_quality = 40
            pdf_settings = '/screen'
        else:
            image_resolution = 50
            image_quality = 20
            pdf_settings = '/screen'
        
        # Method 1: Aggressive PyMuPDF compression with image processing
        if PYMUPDF_AVAILABLE:
            doc = None
            try:
                print("Starting PyMuPDF compression...")
                doc = fitz.open(pdf_path)
                
                # Create a new document for maximum compression
                new_doc = fitz.open()
                
                # Process each page
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Create new page with same dimensions
                    new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
                    
                    # Get all images on the page
                    image_list = page.get_images()
                    
                    # Process and compress images
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        try:
                            # Get image
                            pix = fitz.Pixmap(doc, xref)
                            
                            # Skip if image is too small to compress
                            if pix.width < 50 or pix.height < 50:
                                continue
                            
                            # Convert to RGB if needed
                            if pix.n - pix.alpha < 4:
                                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                                pix = pix1
                            
                            # Calculate new dimensions based on resolution
                            scale_factor = image_resolution / 300  # Assume original is 300 DPI
                            new_width = int(pix.width * scale_factor)
                            new_height = int(pix.height * scale_factor)
                            
                            # Resize image if needed
                            if new_width < pix.width or new_height < pix.height:
                                pix = fitz.Pixmap(pix, fitz.IRect(0, 0, pix.width, pix.height))
                                pix = fitz.Pixmap(pix, fitz.IRect(0, 0, new_width, new_height))
                            
                            # Compress image with aggressive settings
                            img_data = pix.tobytes("jpeg", quality=image_quality)
                            
                            # Replace image in original document
                            doc.update_stream(xref, img_data)
                            
                            pix = None
                        except Exception as e:
                            print(f"Image compression failed: {e}")
                            continue
                    
                    # Copy page content to new document
                    new_page.show_pdf_page(new_page.rect, doc, page_num)
                
                # Save new document with maximum compression
                new_doc.save(output_path, 
                           garbage=4,
                           deflate=True,
                           clean=True,
                           linear=True,
                           pretty=False,
                           ascii=False,
                           compress=True)
                
                new_doc.close()
                doc.close()
                
                # Check compression result
                if os.path.exists(output_path):
                    compressed_size = os.path.getsize(output_path)
                    print(f"PyMuPDF compressed size: {compressed_size} bytes")
                    
                    if compressed_size < original_size:
                        print(f"PyMuPDF compression successful: {((original_size - compressed_size) / original_size * 100):.1f}% reduction")
                        return send_file(
                            output_path,
                            mimetype='application/pdf',
                            as_attachment=True,
                            download_name=f'compressed_{original_filename}'
                        )
                    else:
                        print("PyMuPDF compression didn't reduce size, trying other methods...")
                
            except Exception as e:
                print(f"PyMuPDF compression failed: {e}")
                pass
            finally:
                if doc:
                    try:
                        doc.close()
                    except:
                        pass
        
        # Method 2: Aggressive Ghostscript compression
        print("Trying Ghostscript compression...")
        for gs_command in ['gswin64c', 'gswin32c', 'gs']:
            try:
                # Most aggressive Ghostscript settings
                cmd = [
                    gs_command,
                    '-sDEVICE=pdfwrite',
                    '-dCompatibilityLevel=1.4',
                    '-dPDFSETTINGS=/screen',  # Most aggressive
                    '-dColorImageDownsampleType=/Bicubic',
                    f'-dColorImageResolution={image_resolution}',
                    '-dGrayImageDownsampleType=/Bicubic',
                    f'-dGrayImageResolution={image_resolution}',
                    '-dMonoImageDownsampleType=/Bicubic',
                    f'-dMonoImageResolution={image_resolution}',
                    '-dColorConversionStrategy=/sRGB',
                    '-dEmbedAllFonts=false',
                    '-dSubsetFonts=true',
                    '-dOptimize=true',
                    '-dDownsampleColorImages=true',
                    '-dDownsampleGrayImages=true',
                    '-dDownsampleMonoImages=true',
                    '-dNOPAUSE',
                    '-dQUIET',
                    '-dBATCH',
                    '-sOutputFile=' + output_path + '_gs',
                    pdf_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and os.path.exists(output_path + '_gs'):
                    gs_size = os.path.getsize(output_path + '_gs')
                    print(f"Ghostscript compressed size: {gs_size} bytes")
                    
                    if gs_size < original_size:
                        print(f"Ghostscript compression successful: {((original_size - gs_size) / original_size * 100):.1f}% reduction")
                        # Use Ghostscript result
                        try:
                            if os.path.exists(output_path):
                                os.unlink(output_path)
                            os.rename(output_path + '_gs', output_path)
                        except:
                            pass
                        return send_file(
                            output_path,
                            mimetype='application/pdf',
                            as_attachment=True,
                            download_name=f'compressed_{original_filename}'
                        )
                    else:
                        print("Ghostscript didn't reduce size")
                        try:
                            os.unlink(output_path + '_gs')
                        except:
                            pass
                break
            except Exception as e:
                print(f"Ghostscript {gs_command} failed: {e}")
                continue
        
        # Fallback: Use PyPDF2 with enhanced compression
        pdf_file = None
        output_file = None
        try:
            pdf_file = open(pdf_path, 'rb')
            reader = PyPDF2.PdfReader(pdf_file)
            writer = PyPDF2.PdfWriter()
            
            # Add pages to writer with maximum compression
            for page in reader.pages:
                # Compress the page content streams
                page.compress_content_streams()
                writer.add_page(page)
            
            # Set compression parameters
            writer._compress = True
            
            # Write compressed PDF with maximum compression
            output_file = open(output_path, 'wb')
            writer.write(output_file)
        finally:
            # Always close files
            if pdf_file:
                try:
                    pdf_file.close()
                except:
                    pass
            if output_file:
                try:
                    output_file.close()
                except:
                    pass
        
        # Try additional compression using different methods
        try:
            # Method 1: Try with different Ghostscript commands
            for gs_command in ['gswin64c', 'gswin32c', 'gs']:
                try:
                    # Most aggressive compression settings
                    aggressive_cmd = [
                        gs_command,
                        '-sDEVICE=pdfwrite',
                        '-dCompatibilityLevel=1.4',
                        '-dPDFSETTINGS=/screen',  # Most aggressive
                        '-dColorImageDownsampleType=/Bicubic',
                        '-dColorImageResolution=72',  # Very low resolution
                        '-dGrayImageDownsampleType=/Bicubic',
                        '-dGrayImageResolution=72',
                        '-dMonoImageDownsampleType=/Bicubic',
                        '-dMonoImageResolution=72',
                        '-dNOPAUSE',
                        '-dQUIET',
                        '-dBATCH',
                        '-sOutputFile=' + output_path + '_compressed',
                        output_path
                    ]
                    
                    result = subprocess.run(aggressive_cmd, capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0 and os.path.exists(output_path + '_compressed'):
                        # Replace with more compressed version
                        try:
                            os.unlink(output_path)
                            os.rename(output_path + '_compressed', output_path)
                        except:
                            pass
                        break
                except:
                    continue
        except:
            pass
        
        # Method 3: Ultra-aggressive - Convert to images and back to PDF
        print("Trying ultra-aggressive image conversion method...")
        try:
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(pdf_path)
                
                # Create new document
                new_doc = fitz.open()
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Convert page to image with low resolution
                    mat = fitz.Matrix(1.0, 1.0)  # Scale factor
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    # Convert to PIL Image for compression
                    img_data = pix.tobytes("ppm")
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Resize image based on quality
                    if quality < 40:
                        # Very aggressive resizing for low quality
                        new_width = int(img.width * 0.5)
                        new_height = int(img.height * 0.5)
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Save compressed image
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='JPEG', quality=image_quality, optimize=True)
                    img_buffer.seek(0)
                    
                    # Create new page with image
                    new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
                    new_page.insert_image(new_page.rect, stream=img_buffer.getvalue())
                    
                    pix = None
                    img = None
                
                # Save with maximum compression
                new_doc.save(output_path + '_ultra',
                           garbage=4,
                           deflate=True,
                           clean=True,
                           linear=True,
                           pretty=False,
                           ascii=False,
                           compress=True)
                
                new_doc.close()
                doc.close()
                
                # Check if ultra compression worked
                if os.path.exists(output_path + '_ultra'):
                    ultra_size = os.path.getsize(output_path + '_ultra')
                    print(f"Ultra-aggressive compressed size: {ultra_size} bytes")
                    
                    if ultra_size < original_size:
                        print(f"Ultra-aggressive compression successful: {((original_size - ultra_size) / original_size * 100):.1f}% reduction")
                        try:
                            if os.path.exists(output_path):
                                os.unlink(output_path)
                            os.rename(output_path + '_ultra', output_path)
                        except:
                            pass
                        return send_file(
                            output_path,
                            mimetype='application/pdf',
                            as_attachment=True,
                            download_name=f'compressed_{original_filename}'
                        )
                    else:
                        print("Ultra-aggressive didn't reduce size")
                        try:
                            os.unlink(output_path + '_ultra')
                        except:
                            pass
        except Exception as e:
            print(f"Ultra-aggressive compression failed: {e}")
            pass
        
        # Return the compressed PDF
        return send_file(
            output_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'compressed_{original_filename}'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/protect-pdf', methods=['POST'])
def protect_pdf_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    password = request.form.get('password', '')
    
    if not password:
        return jsonify({'error': 'Password required'}), 400
    
    try:
        # Read the PDF
        reader = PyPDF2.PdfReader(file)
        writer = PyPDF2.PdfWriter()
        
        # Add pages to writer
        for page in reader.pages:
            writer.add_page(page)
        
        # Encrypt the PDF
        writer.encrypt(password)
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
            output_path = tmp_pdf.name
        
        # Write protected PDF
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        # Return the protected PDF
        return send_file(
            output_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'protected_{secure_filename(file.filename)}'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/remove-pdf-password', methods=['POST'])
def remove_pdf_password_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    password = request.form.get('password', '')
    
    try:
        # Read the PDF with password
        reader = PyPDF2.PdfReader(file)
        if reader.is_encrypted:
            if not password:
                return jsonify({'error': 'Password required to unlock PDF'}), 400
            reader.decrypt(password)
        
        writer = PyPDF2.PdfWriter()
        
        # Add pages to writer
        for page in reader.pages:
            writer.add_page(page)
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
            output_path = tmp_pdf.name
        
        # Write unprotected PDF
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        # Return the unprotected PDF
        return send_file(
            output_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'unlocked_{secure_filename(file.filename)}'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/edit-pdf', methods=['POST'])
def edit_pdf_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    mode = request.form.get('mode', 'text')
    
    try:
        # For now, return the original PDF (edit functionality would be more complex)
        # This is a placeholder implementation
        reader = PyPDF2.PdfReader(file)
        writer = PyPDF2.PdfWriter()
        
        for page in reader.pages:
            writer.add_page(page)
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
            output_path = tmp_pdf.name
        
        # Write PDF
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        # Return the PDF
        return send_file(
            output_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'edited_{secure_filename(file.filename)}'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/video-formats', methods=['POST'])
def api_video_formats():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    ydl_opts = {'quiet': True, 'skip_download': True, 'forcejson': True, 'extract_flat': False}
    # Map of standard p-style labels to accepted WxH, notes, and heights
    resolution_map = {
        '144p': ['144p', '256x144', 'height=144'],
        '240p': ['240p', '426x240', 'height=240'],
        '360p': ['360p', '640x360', 'height=360'],
        '480p': ['480p', '854x480', 'height=480'],
        '720p': ['720p', '1280x720', 'hd720', 'height=720'],
        '1080p': ['1080p', '1920x1080', 'hd1080', 'height=1080'],
        '2160p': ['2160p', '3840x2160', 'hd2160', '4k', 'uhd', 'height=2160']
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        formats_by_res = {k: [] for k in resolution_map}
        for f in info.get('formats', []):
            if not (f.get('url') and f.get('ext') == 'mp4'):
                continue
            res = (f.get('resolution') or f.get('format_note') or '').lower()
            height = str(f.get('height') or '').strip()
            width = str(f.get('width') or '').strip()
            wxh = f'{width}x{height}' if width and height else ''
            for label, keys in resolution_map.items():
                for key in keys:
                    if key == res:
                        formats_by_res[label].append(f)
                    elif key.startswith('height=') and height and key.split('=')[1] == height:
                        formats_by_res[label].append(f)
                    elif key == wxh:
                        formats_by_res[label].append(f)
                    elif key == f.get('format_note', '').lower():
                        formats_by_res[label].append(f)
        # For each label, pick the best format (largest filesize or bitrate)
        formats = []
        for label, flist in formats_by_res.items():
            if not flist:
                continue
            # Prefer largest filesize, then highest tbr (bitrate)
            best = max(flist, key=lambda x: (x.get('filesize') or 0, x.get('tbr') or 0))
            formats.append({
                'format_id': best.get('format_id'),
                'ext': best.get('ext'),
                'resolution': label,  # always use the p-style label
                'format_note': best.get('format_note'),
                'filesize': best.get('filesize') or best.get('filesize_approx'),
            })
        # Sort by resolution order
        label_order = list(resolution_map.keys())
        formats.sort(key=lambda x: label_order.index(x['resolution']))
        # Add video info: title, thumbnail, duration
        video_info = {
            'title': info.get('title', ''),
            'thumbnail': info.get('thumbnail', ''),
            'duration': info.get('duration', 0)
        }
        return jsonify({'formats': formats, **video_info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/video-download')
def api_video_download():
    url = request.args.get('url')
    format_id = request.args.get('format_id')
    if not url or not format_id:
        return 'Missing parameters', 400
    try:
        # First, get the video title using yt-dlp
        try:
            ydl_opts = {'quiet': True, 'skip_download': True, 'forcejson': True, 'extract_flat': False}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'video')
        except Exception:
            title = 'video'
        safe_title = ''.join(c for c in title if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
        filename = f"{safe_title}.mp4"
        # Use yt-dlp to stream video directly to client (no temp file)
        cmd = [
            'yt-dlp',
            '-f', format_id,
            '-o', '-',  # output to stdout
            url
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        def generate():
            while True:
                chunk = p.stdout.read(8192)
                if not chunk:
                    break
                yield chunk
            p.stdout.close()
            p.stderr.close()
            p.wait()
        headers = {
            'Content-Disposition': f'attachment; filename="{filename}"',
            'Content-Type': 'video/mp4'
        }
        return Response(generate(), headers=headers)
    except Exception as e:
        return f'Error: {e}', 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 