#!/usr/bin/env python3
import os
import sys
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def build_pdf(md_path, pdf_path):
    # Register STHeiti Chinese font and its bold partner
    font_path = '/System/Library/Fonts/STHeiti Light.ttc'
    bold_font_path = '/System/Library/Fonts/STHeiti Medium.ttc'
    
    if os.path.exists(font_path) and os.path.exists(bold_font_path):
        pdfmetrics.registerFont(TTFont('STHeiti', font_path))
        pdfmetrics.registerFont(TTFont('STHeiti-Bold', bold_font_path))
        pdfmetrics.registerFontFamily('STHeiti', normal='STHeiti', bold='STHeiti-Bold')
        body_font = 'STHeiti'
        # Use STHeiti for code block font to support Chinese comments
        code_font = 'STHeiti'
    elif os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont('STHeiti', font_path))
        body_font = 'STHeiti'
        code_font = 'STHeiti'
    else:
        body_font = 'Helvetica'
        code_font = 'Courier'

    # Read markdown file
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Create document
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom styles to support Chinese and better aesthetics
    styles.add(ParagraphStyle(
        name='MainTitle',
        fontName=body_font,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1A202C'),
        spaceAfter=15,
        alignment=0
    ))
    styles.add(ParagraphStyle(
        name='Heading2_Custom',
        fontName=body_font,
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2D3748'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        name='Heading3_Custom',
        fontName=body_font,
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#4A5568'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        name='BodyText_Custom',
        fontName=body_font,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name='Bullet_Custom',
        fontName=body_font,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#2D3748'),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        name='Code_Custom',
        fontName=code_font,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#2D3748'),
        backColor=colors.HexColor('#F7FAFC'),
        borderColor=colors.HexColor('#E2E8F0'),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name='Code_Monospace',
        fontName='Courier',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#2D3748'),
        backColor=colors.HexColor('#F7FAFC'),
        borderColor=colors.HexColor('#E2E8F0'),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=8
    ))

    flowables = []
    in_code_block = False
    code_content = []
    
    for line in lines:
        stripped = line.strip()
        
        # Check code block delimiters
        if stripped.startswith('```'):
            if in_code_block:
                # End of code block, append preformatted block
                code_text = ''.join(code_content)
                # If there are Chinese characters, use custom font (STHeiti) to prevent tofu blocks;
                # otherwise use Courier (monospace) so ASCII diagrams align perfectly.
                has_chinese = bool(re.search(r'[\u4e00-\u9fff]', code_text))
                selected_style = styles['Code_Custom'] if has_chinese else styles['Code_Monospace']
                flowables.append(Preformatted(code_text, selected_style))
                code_content = []
                in_code_block = False
            else:
                in_code_block = True
            continue
            
        if in_code_block:
            code_content.append(line)
            continue

        # Markdown headings
        if stripped.startswith('# '):
            title_text = stripped[2:]
            flowables.append(Paragraph(title_text, styles['MainTitle']))
            flowables.append(Spacer(1, 10))
            continue
        elif stripped.startswith('## '):
            h2_text = stripped[3:]
            flowables.append(Paragraph(h2_text, styles['Heading2_Custom']))
            continue
        elif stripped.startswith('### '):
            h3_text = stripped[4:]
            flowables.append(Paragraph(h3_text, styles['Heading3_Custom']))
            continue
            
        # Horizontal rule
        if stripped == '---':
            # Add thin divider line using a table with no content but borders
            divider = Table([['']], colWidths=[504], rowHeights=[1])
            divider.setStyle(TableStyle([
                ('LINEABOVE', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 0),
                ('TOPPADDING', (0,0), (-1,-1), 0),
            ]))
            flowables.append(Spacer(1, 6))
            flowables.append(divider)
            flowables.append(Spacer(1, 6))
            continue
            
        # Bullet list items
        if stripped.startswith('* ') or stripped.startswith('- '):
            bullet_text = stripped[2:]
            # Replace markdown inline styles like `code`, **bold** to simple HTML formatting
            bullet_text = re.sub(r'`([^`]+)`', r'<font face="Courier">\1</font>', bullet_text)
            bullet_text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', bullet_text)
            bullet_text = bullet_text.replace(r'$\to$', ' -> ')
            flowables.append(Paragraph('- ' + bullet_text, styles['Bullet_Custom']))
            continue
            
        if stripped.startswith('1. ') or stripped.startswith('2. ') or stripped.startswith('3. ') or stripped.startswith('4. ') or stripped.startswith('5. '):
            list_text = stripped
            list_text = re.sub(r'`([^`]+)`', r'<font face="Courier">\1</font>', list_text)
            list_text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', list_text)
            list_text = list_text.replace(r'$\to$', ' -> ')
            flowables.append(Paragraph(list_text, styles['Bullet_Custom']))
            continue

        # Regular text
        if stripped:
            text = stripped
            # Inline formatting replacements
            text = re.sub(r'`([^`]+)`', r'<font face="Courier">\1</font>', text)
            text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
            text = text.replace(r'$\to$', ' -> ')
            flowables.append(Paragraph(text, styles['BodyText_Custom']))
        else:
            flowables.append(Spacer(1, 4))

    # Build PDF
    doc.build(flowables)
    print("PDF build complete!")

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        md = sys.argv[1]
        pdf = sys.argv[2]
    else:
        md = '/Users/linwanyi/Library/Mobile Documents/com~apple~CloudDocs/Codex/PvrGPU/todo/Geometry.md'
        pdf = '/Users/linwanyi/Library/Mobile Documents/com~apple~CloudDocs/Codex/PvrGPU/todo/Geometry.pdf'
    build_pdf(md, pdf)
