# 7.4 Solutions - Professional Image Tools

A modern, responsive Flask web application that provides professional image editing tools with a beautiful UI inspired by the Crypto KuCoin P2P Market design.

## Features

- **Image Compression**: Reduce file sizes while maintaining quality
- **Image Resizing**: Resize images to any dimension with precision
- **Format Conversion**: Convert between JPG, PNG, GIF, WebP, and more
- **Image Cropping**: Crop images to any aspect ratio or custom dimensions
- **Image Rotation**: Rotate images by any angle
- **Watermarking**: Add text or image watermarks with full control
- **100% Secure**: All processing done locally in the browser
- **Mobile Friendly**: Responsive design that works on all devices
- **Free Forever**: No hidden costs or limitations

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Custom CSS with modern design principles
- **Icons**: Font Awesome
- **Fonts**: Inter (Google Fonts)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd 7.4-solutions
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Project Structure

```
7.4-solutions/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── templates/            # HTML templates
    ├── base.html         # Base template with common layout
    ├── index.html        # Homepage
    ├── compress.html     # Image compression tool
    ├── resize.html       # Image resizing tool
    ├── convert.html      # Format conversion tool
    ├── crop.html         # Image cropping tool
    ├── rotate.html       # Image rotation tool
    ├── watermark.html    # Watermarking tool
    ├── about.html        # About page
    └── contact.html      # Contact page
```

## Pages and Features

### Homepage (`/`)
- Hero section with call-to-action
- Grid of available tools
- Features and benefits section

### Image Tools
Each tool page includes:
- Drag-and-drop file upload
- Tool-specific options and settings
- Real-time file processing simulation
- Download functionality

#### Compress Images (`/compress`)
- File size reduction with quality preservation
- Multiple file support
- Compression ratio display

#### Resize Images (`/resize`)
- Custom width/height input
- Preset aspect ratios (16:9, 4:3, 1:1, etc.)
- Maintain aspect ratio option

#### Convert Images (`/convert`)
- Support for JPG, PNG, GIF, WebP, BMP, TIFF
- Batch conversion
- Quality preservation

#### Crop Images (`/crop`)
- Free crop and preset aspect ratios
- Custom dimensions
- Visual crop preview

#### Rotate Images (`/rotate`)
- 90°, 180°, 270° presets
- Custom angle input
- Orientation correction

#### Add Watermark (`/watermark`)
- Text and image watermarks
- Customizable position, size, opacity
- Multiple positioning options

### About Page (`/about`)
- Company mission and values
- Tool overview
- Statistics and achievements

### Contact Page (`/contact`)
- Contact form with validation
- Company information
- FAQ section

## Design Features

- **Modern Dark Theme**: Inspired by Crypto KuCoin P2P Market design
- **Gradient Accents**: Cyan/blue gradient for highlights
- **Glass Morphism**: Translucent cards and overlays
- **Smooth Animations**: Hover effects and transitions
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Accessibility**: High contrast and readable typography

## Browser Compatibility

- Chrome (recommended)
- Firefox
- Safari
- Edge

## Development

### Adding New Tools

1. Add a new route in `app.py`:
   ```python
   @app.route('/new-tool')
   def new_tool():
       return render_template('new-tool.html')
   ```

2. Create a new template file `templates/new-tool.html`
3. Add the tool to the navigation in `base.html`
4. Update the homepage tools grid

### Customizing Styles

All styles are inlined in the HTML templates for simplicity. To modify the design:

1. Edit the `<style>` section in `base.html` for global styles
2. Modify individual page styles in their respective template files
3. Update color scheme by changing the CSS custom properties

## Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
For production deployment, consider:

1. **WSGI Server**: Use Gunicorn or uWSGI
2. **Reverse Proxy**: Nginx or Apache
3. **Environment Variables**: Set `FLASK_ENV=production`
4. **HTTPS**: SSL certificate for security

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For support, questions, or feature requests:
- Email: support@7.4solutions.com
- Website: www.7.4solutions.com

---

**7.4 Solutions** - Empowering creators with professional image tools since 2024. 