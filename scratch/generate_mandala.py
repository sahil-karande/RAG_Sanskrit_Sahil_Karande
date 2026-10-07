import math

def make_mandala_svg():
    lines = []
    lines.append("<svg xmlns='http://www.w3.org/2000/svg' viewBox='-350 -350 700 700' width='100%' height='100%'>\n")
    lines.append("  <g fill='none' stroke='#8b572a' stroke-linecap='round' stroke-linejoin='round'>\n")
    
    # Center bindu & rings
    lines.append("    <circle cx='0' cy='0' r='10' fill='#8b572a' stroke='none'/>\n")
    lines.append("    <circle cx='0' cy='0' r='28' stroke-width='2'/>\n")
    lines.append("    <circle cx='0' cy='0' r='48' stroke-width='1.5' stroke-dasharray='4,4'/>\n")
    lines.append("    <circle cx='0' cy='0' r='75' stroke-width='2'/>\n")
    
    # 8 inner lotus petals (Ashta-Dala Padma)
    for i in range(8):
        angle = i * (math.pi / 4)
        r_inner = 75
        r_outer = 145
        tip_x = r_outer * math.cos(angle)
        tip_y = r_outer * math.sin(angle)
        base1_x = r_inner * math.cos(angle - 0.3)
        base1_y = r_inner * math.sin(angle - 0.3)
        base2_x = r_inner * math.cos(angle + 0.3)
        base2_y = r_inner * math.sin(angle + 0.3)
        ctrl1_x = (r_outer * 0.82) * math.cos(angle - 0.22)
        ctrl1_y = (r_outer * 0.82) * math.sin(angle - 0.22)
        ctrl2_x = (r_outer * 0.82) * math.cos(angle + 0.22)
        ctrl2_y = (r_outer * 0.82) * math.sin(angle + 0.22)
        lines.append(f"    <path d='M {base1_x:.1f},{base1_y:.1f} Q {ctrl1_x:.1f},{ctrl1_y:.1f} {tip_x:.1f},{tip_y:.1f} Q {ctrl2_x:.1f},{ctrl2_y:.1f} {base2_x:.1f},{base2_y:.1f}' stroke-width='2'/>\n")
        lines.append(f"    <circle cx='{tip_x:.1f}' cy='{tip_y:.1f}' r='3.5' fill='#8b572a' stroke='none'/>\n")

    # Intermediate rings
    lines.append("    <circle cx='0' cy='0' r='165' stroke-width='2.5'/>\n")
    lines.append("    <circle cx='0' cy='0' r='180' stroke-width='1.5' stroke-dasharray='4,4'/>\n")

    # 16 radiant petals
    for i in range(16):
        angle = i * (math.pi / 8)
        tip_x = 235 * math.cos(angle)
        tip_y = 235 * math.sin(angle)
        base1_x = 180 * math.cos(angle - 0.16)
        base1_y = 180 * math.sin(angle - 0.16)
        base2_x = 180 * math.cos(angle + 0.16)
        base2_y = 180 * math.sin(angle + 0.16)
        lines.append(f"    <path d='M {base1_x:.1f},{base1_y:.1f} L {tip_x:.1f},{tip_y:.1f} L {base2_x:.1f},{base2_y:.1f}' stroke-width='1.5'/>\n")
        lines.append(f"    <circle cx='{tip_x:.1f}' cy='{tip_y:.1f}' r='2.5' fill='#8b572a' stroke='none'/>\n")

    # Outer boundary rings & 32 bead pearls (Bindu Mala)
    lines.append("    <circle cx='0' cy='0' r='255' stroke-width='2'/>\n")
    lines.append("    <circle cx='0' cy='0' r='275' stroke-width='2.5'/>\n")
    for i in range(32):
        angle = i * (math.pi / 16)
        dx = 265 * math.cos(angle)
        dy = 265 * math.sin(angle)
        lines.append(f"    <circle cx='{dx:.1f}' cy='{dy:.1f}' r='2' fill='#8b572a' stroke='none'/>\n")

    lines.append("  </g>\n")
    lines.append("</svg>")
    return ''.join(lines)

svg_str = make_mandala_svg()
with open('assets/sacred_mandala_watermark.svg', 'w', encoding='utf-8') as f:
    f.write(svg_str)
print('Successfully generated assets/sacred_mandala_watermark.svg!')
