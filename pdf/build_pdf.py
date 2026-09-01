#!/usr/bin/env python3
import os
import sys
import re
import html
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def build_pdf(md_path, pdf_path):
    # Register fonts
    font_path = '/System/Library/Fonts/STHeiti Light.ttc'
    bold_font_path = '/System/Library/Fonts/STHeiti Medium.ttc'

    if os.path.exists(font_path) and os.path.exists(bold_font_path):
        pdfmetrics.registerFont(TTFont('STHeiti', font_path))
        pdfmetrics.registerFont(TTFont('STHeiti-Bold', bold_font_path))
        pdfmetrics.registerFontFamily('STHeiti', normal='STHeiti', bold='STHeiti-Bold')
        body_font = 'STHeiti'
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
        content = f.read()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    content_width = 532 # 612 - 80

    styles.add(ParagraphStyle(
        name='MainTitle',
        fontName=body_font,
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=12,
        alignment=0
    ))
    styles.add(ParagraphStyle(
        name='Heading2_Custom',
        fontName=body_font,
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        name='Heading3_Custom',
        fontName=body_font,
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#2D3748'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        name='BodyText_Custom',
        fontName=body_font,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        name='Bullet_Custom',
        fontName=body_font,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#2D3748'),
        leftIndent=12,
        spaceAfter=3
    ))
    styles.add(ParagraphStyle(
        name='Code_Custom',
        fontName=code_font,
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#2D3748'),
        backColor=colors.HexColor('#F7FAFC'),
        borderColor=colors.HexColor('#E2E8F0'),
        borderWidth=0.5,
        borderPadding=5,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name='Code_Monospace',
        fontName='Courier',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#2D3748'),
        backColor=colors.HexColor('#F7FAFC'),
        borderColor=colors.HexColor('#E2E8F0'),
        borderWidth=0.5,
        borderPadding=5,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name='TableHeader',
        fontName=body_font,
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=0
    ))
    styles.add(ParagraphStyle(
        name='TableCell',
        fontName=body_font,
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#2D3748'),
        alignment=0
    ))
    styles.add(ParagraphStyle(
        name='DiagramStepNum',
        fontName=body_font,
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#2B6CB0'),
    ))
    styles.add(ParagraphStyle(
        name='DiagramStepBody',
        fontName=body_font,
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#2D3748'),
    ))
    styles.add(ParagraphStyle(
        name='DiagramArrow',
        fontName=body_font,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#4A5568'),
        alignment=1
    ))

    def format_inline(raw):
        parts = []
        last_end = 0
        for m in re.finditer(r'`([^`]+)`', raw):
            plain = raw[last_end:m.start()]
            plain = html.escape(plain)
            plain = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', plain)
            plain = plain.replace(r'$\to$', ' &rarr; ').replace(r'->', ' &rarr; ')
            parts.append(plain)
            code_val = html.escape(m.group(1))
            parts.append(f'<font face="Courier" color="#C53030">{code_val}</font>')
            last_end = m.end()
        rem = raw[last_end:]
        rem = html.escape(rem)
        rem = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', rem)
        rem = rem.replace(r'$\to$', ' &rarr; ').replace(r'->', ' &rarr; ')
        parts.append(rem)
        return ''.join(parts)

    def parse_flowchart_blocks(lines):
        """Parse ASCII box diagrams into elegant ReportLab Card Tables"""
        # Check if horizontal multi-column boxes (like in Section 6)
        first_line = lines[0].strip() if lines else ''
        if first_line.count('+--') >= 2:
            # Horizontal 3-column stage flow
            col1, col2, col3 = [], [], []
            for line in lines:
                s = line.strip()
                if s.startswith('+--'):
                    continue
                # split by |
                parts = [p.strip() for p in line.split('|')]
                # parts[1] is col1, parts[3] is col2, parts[5] is col3
                if len(parts) >= 7:
                    if parts[1]: col1.append(parts[1])
                    if parts[3]: col2.append(parts[3])
                    if parts[5]: col3.append(parts[5])

            def make_stage_card(items):
                title = items[0] if items else ''
                body = items[1:] if len(items) > 1 else []
                cell_f = [Paragraph(f'<b>{format_inline(title)}</b>', styles['DiagramStepNum'])]
                for b in body:
                    cell_f.append(Paragraph(format_inline(b), styles['TableCell']))
                ct = Table([[cell_f]], colWidths=[150])
                ct.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F7FAFC')),
                    ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E0')),
                    ('PADDING', (0,0), (-1,-1), 5),
                ]))
                return ct

            arrow_p = Paragraph('<font size="12" color="#2B6CB0">&rarr;</font>', styles['DiagramArrow'])
            h_table = Table([[make_stage_card(col1), arrow_p, make_stage_card(col2), arrow_p, make_stage_card(col3)]],
                            colWidths=[155, 30, 155, 30, 155])
            h_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('PADDING', (0,0), (-1,-1), 2),
            ]))
            return h_table

        # Vertical flowchart
        boxes = []
        current_box = []
        in_box = False
        current_arrow = []

        for line in lines:
            s = line.strip()
            if s.startswith('+--') or s.startswith('+=='):
                if in_box:
                    if current_box:
                        boxes.append(('box', '\n'.join(current_box)))
                        current_box = []
                    in_box = False
                else:
                    if current_arrow:
                        arrow_text = ' '.join(current_arrow).strip()
                        if arrow_text:
                            boxes.append(('arrow', arrow_text))
                        current_arrow = []
                    in_box = True
            elif in_box:
                if s.startswith('|') and s.endswith('|'):
                    content_line = s[1:-1].strip()
                    if content_line:
                        current_box.append(content_line)
                elif s.startswith('|'):
                    content_line = s[1:].strip()
                    if content_line:
                        current_box.append(content_line)
            else:
                # outside box, strip leading and trailing vertical bars or standalone arrow markers
                clean_s = re.sub(r'^[|\sv\^]+', '', s).strip()
                if clean_s:
                    current_arrow.append(clean_s)

        if not boxes:
            return None

        # Build Flowable Table
        table_rows = []
        for b_type, b_content in boxes:
            if b_type == 'box':
                lines_in_b = b_content.split('\n')
                title_line = lines_in_b[0]
                body_lines = lines_in_b[1:]

                cell_flowables = [
                    Paragraph(f'<b>{format_inline(title_line)}</b>', styles['DiagramStepNum'])
                ]
                for bl in body_lines:
                    cell_flowables.append(Paragraph(format_inline(bl), styles['DiagramStepBody']))

                card_table = Table([[cell_flowables]], colWidths=[content_width - 20])
                card_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F7FAFC')),
                    ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E0')),
                    ('PADDING', (0,0), (-1,-1), 5),
                    ('TOPPADDING', (0,0), (-1,-1), 4),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ]))
                table_rows.append([card_table])
            elif b_type == 'arrow':
                arrow_p = Paragraph(f'&darr; <font color="#4A5568"><b>{format_inline(b_content)}</b></font>', styles['DiagramArrow'])
                table_rows.append([arrow_p])

        flowchart_table = Table(table_rows, colWidths=[content_width])
        flowchart_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 2),
        ]))
        return flowchart_table

    lines = content.split('\n')
    flowables = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 1. Code blocks or diagrams
        if stripped.startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1 # skip closing ```

            # Check if this code block is an ASCII box diagram
            has_ascii_box = any(l.strip().startswith('+--') or l.strip().startswith('+==') for l in code_lines)
            if has_ascii_box:
                flowchart = parse_flowchart_blocks(code_lines)
                if flowchart:
                    flowables.append(Spacer(1, 4))
                    flowables.append(flowchart)
                    flowables.append(Spacer(1, 6))
                    continue

            # Regular code block
            raw_code = '\n'.join(code_lines)
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', raw_code))
            selected_style = styles['Code_Custom'] if has_chinese else styles['Code_Monospace']
            flowables.append(Preformatted(raw_code.rstrip(), selected_style))
            flowables.append(Spacer(1, 4))
            continue

        # 2. Markdown Tables
        if stripped.startswith('|') and stripped.endswith('|') and '|' in stripped[1:-1]:
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|') and lines[i].strip().endswith('|'):
                table_lines.append(lines[i].strip())
                i += 1

            # Parse table
            if len(table_lines) >= 2:
                header_raw = [c.strip() for c in table_lines[0].split('|')[1:-1]]
                # check separator
                start_row = 2 if len(table_lines) > 1 and re.match(r'^[|\s:-]+$', table_lines[1]) else 1

                col_count = len(header_raw)
                table_data = []

                # Header row
                header_row = [Paragraph(f'<b>{format_inline(c)}</b>', styles['TableHeader']) for c in header_raw]
                table_data.append(header_row)

                # Data rows
                for row_line in table_lines[start_row:]:
                    row_cells = [c.strip() for c in row_line.split('|')[1:-1]]
                    # pad if missing
                    while len(row_cells) < col_count:
                        row_cells.append('')
                    row_cells = row_cells[:col_count]
                    table_data.append([Paragraph(format_inline(c), styles['TableCell']) for c in row_cells])

                col_w = content_width / col_count
                t = Table(table_data, colWidths=[col_w] * col_count)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
                    ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')]),
                    ('PADDING', (0,0), (-1,-1), 4),
                    ('TOPPADDING', (0,0), (-1,-1), 4),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ]))
                flowables.append(Spacer(1, 4))
                flowables.append(t)
                flowables.append(Spacer(1, 6))
                continue

        # 3. Headings
        if stripped.startswith('# '):
            flowables.append(Paragraph(stripped[2:], styles['MainTitle']))
            flowables.append(Spacer(1, 6))
            i += 1
            continue
        elif stripped.startswith('## '):
            flowables.append(Paragraph(stripped[3:], styles['Heading2_Custom']))
            i += 1
            continue
        elif stripped.startswith('### '):
            flowables.append(Paragraph(stripped[4:], styles['Heading3_Custom']))
            i += 1
            continue

        # 4. Divider
        if stripped == '---':
            divider = Table([['']], colWidths=[content_width], rowHeights=[1])
            divider.setStyle(TableStyle([
                ('LINEABOVE', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 0),
                ('TOPPADDING', (0,0), (-1,-1), 0),
            ]))
            flowables.append(Spacer(1, 4))
            flowables.append(divider)
            flowables.append(Spacer(1, 4))
            i += 1
            continue

        # 5. List items
        if stripped.startswith('* ') or stripped.startswith('- '):
            bullet_text = format_inline(stripped[2:])
            flowables.append(Paragraph('&bull; ' + bullet_text, styles['Bullet_Custom']))
            i += 1
            continue

        if re.match(r'^\d+\.\s+', stripped):
            num_match = re.match(r'^(\d+\.\s+)(.*)$', stripped)
            list_num = num_match.group(1)
            list_text = format_inline(num_match.group(2))
            flowables.append(Paragraph(f'<b>{list_num}</b>' + list_text, styles['Bullet_Custom']))
            i += 1
            continue

        # 6. Regular text
        if stripped:
            text = format_inline(stripped)
            flowables.append(Paragraph(text, styles['BodyText_Custom']))
        else:
            flowables.append(Spacer(1, 2))

        i += 1

    doc.build(flowables)
    print("PDF build complete!")

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        md = sys.argv[1]
        pdf = sys.argv[2]
    else:
        pdf_dir = os.path.dirname(os.path.abspath(__file__))
        md = os.path.join(pdf_dir, 'driver.md')
        pdf = os.path.join(pdf_dir, 'driver.pdf')
    build_pdf(md, pdf)
