import qrcode
from PIL import Image, ImageDraw

# pip install qrcode[pil]

# URL of the GitHub documentation
url = 'https://mbirjax.readthedocs.io'

# Path to the logo image
logo_path = '../../docs/source/_static/logo.png'

# Generate QR code
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)
qr.add_data(url)
qr.make(fit=True)

# Create an image from the QR code instance
img_qr = qr.make_image(fill='black', back_color='white').convert('RGBA')

# Open the logo image
logo = Image.open(logo_path).convert('RGBA')

# Resize the logo
logo_width = 400
logo_ratio = logo_width / float(logo.size[0])
logo_height = int(float(logo.size[1]) * logo_ratio)
logo = logo.resize((logo_width, logo_height), Image.LANCZOS)

# Calculate the dimensions of the final image
qr_width, qr_height = img_qr.size
padding = 0
total_height = qr_height + logo_height + padding  # Add some space between QR and logo

# Create a new image with an off-white background
bg_color = (235, 235, 230, 255)  # Off-white with full opacity
final_image = Image.new('RGBA', (qr_width, total_height), bg_color)

# Paste the QR code onto the final image
final_image.paste(img_qr, (0, 0), mask=img_qr)

# Paste the logo onto the final image
logo_pos = ((qr_width - logo_width) // 2, qr_height + padding)  # Reduced padding
final_image.paste(logo, logo_pos, mask=logo)

# Create rounded corners mask
radius = 30  # Adjust the radius as needed
mask = Image.new('L', final_image.size, 0)
draw = ImageDraw.Draw(mask)
draw.rounded_rectangle([(0, 0), final_image.size], radius, fill=255)

# Apply rounded corners mask to the final image
rounded_final_image = Image.new('RGBA', final_image.size, (0, 0, 0, 0))
rounded_final_image.paste(final_image, (0, 0), mask=mask)

# Save the final image
rounded_final_image.save('mbirjax-qr-code.png')