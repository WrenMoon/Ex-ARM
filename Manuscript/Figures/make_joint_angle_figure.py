"""Draw the physical joint-angle comparison used in the manuscript."""

import csv
from pathlib import Path
from statistics import mean

from reportlab.lib.colors import Color, HexColor
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "Data" / "Proprioception" / "Physical_Only" / "Common Household Objects" / "trials"
OUTPUT = HERE / "joint_angles_bottle_mug.pdf"
OBJECTS = (("water bottle", "Water bottle", HexColor("#1D5A80")),
           ("mug", "Mug", HexColor("#A54F32")))


def trials_for(name):
    trials = []
    for path in sorted((DATA / name).glob("*/grasp_results.csv"), key=lambda p: int(p.parent.name)):
        with path.open(newline="") as file:
            rows = list(csv.reader(file))
        angles = [float(value) for value in rows[1]]
        if len(angles) != 48:
            raise ValueError(f"Expected 48 angles in {path}")
        trials.append(angles)
    if len(trials) != 5:
        raise ValueError(f"Expected five physical trials for {name}; got {len(trials)}")
    return trials


def draw_panel(pdf, angles, color, grip, x, y, width, height, ymin, ymax):
    left, right, bottom, top = 29, 7, 20, 18
    px, py = x + left, y + bottom
    pw, ph = width - left - right, height - bottom - top

    pdf.setFont("Helvetica", 8)
    for value in range(ymin, ymax + 1, 40):
        yy = py + ph * (value - ymin) / (ymax - ymin)
        pdf.setStrokeColor(HexColor("#E1E4E8"))
        pdf.setLineWidth(0.4)
        pdf.line(px, yy, px + pw, yy)
        pdf.setFillColor(HexColor("#4C5964"))
        pdf.drawRightString(px - 3, yy - 2, str(value))

    for index in (4, 8, 12):
        xx = px + pw * (index - 0.5) / 15
        pdf.setDash(2, 2)
        pdf.setStrokeColor(HexColor("#BAC1C7"))
        pdf.line(xx, py, xx, py + ph)
        pdf.setDash()

    def point(index, angle):
        return px + pw * index / 15, py + ph * (angle - ymin) / (ymax - ymin)

    for trial in angles:
        pdf.setStrokeColor(Color(color.red, color.green, color.blue, alpha=0.27))
        pdf.setLineWidth(0.7)
        values = trial[grip * 16:(grip + 1) * 16]
        path = pdf.beginPath()
        path.moveTo(*point(0, values[0]))
        for index, value in enumerate(values[1:], start=1):
            path.lineTo(*point(index, value))
        pdf.drawPath(path)

    averages = [mean(trial[grip * 16 + index] for trial in angles) for index in range(16)]
    pdf.setStrokeColor(color)
    pdf.setLineWidth(1.8)
    path = pdf.beginPath()
    path.moveTo(*point(0, averages[0]))
    for index, value in enumerate(averages[1:], start=1):
        path.lineTo(*point(index, value))
    pdf.drawPath(path)
    pdf.setFillColor(color)
    for index, value in enumerate(averages):
        xx, yy = point(index, value)
        pdf.circle(xx, yy, 1.35, fill=1, stroke=0)

    pdf.setStrokeColor(HexColor("#76818A"))
    pdf.setLineWidth(0.5)
    pdf.line(px, py, px + pw, py)
    pdf.line(px, py, px, py + ph)
    pdf.setFillColor(HexColor("#4C5964"))
    for index in (0, 4, 8, 12, 15):
        xx = px + pw * index / 15
        pdf.drawCentredString(xx, py - 10, str(index))


def main():
    series = {name: trials_for(name) for name, _, _ in OBJECTS}
    page_width, page_height = 7.25 * inch, 5.8 * inch
    pdf = canvas.Canvas(str(OUTPUT), pagesize=(page_width, page_height))
    pdf.setTitle("Final joint angles for water bottle and mug physical trials")
    pdf.setAuthor("Ex-ARM physical-only proprioception study")

    panel_width, panel_height = 3.18 * inch, 1.625 * inch
    left_x, right_x = 42, page_width - panel_width - 10
    top_y = page_height - 27
    pdf.setFont("Helvetica-Bold", 10)
    for x, (_, label, color) in zip((left_x, right_x), OBJECTS):
        pdf.setFillColor(color)
        pdf.drawCentredString(x + panel_width / 2, page_height - 16, label)

    for grip in range(3):
        values = [trial[grip * 16 + index]
                  for name, _, _ in OBJECTS for trial in series[name] for index in range(16)]
        ymin = int(min(values) // 40 * 40)
        ymax = int((max(values) + 39) // 40 * 40)
        if ymax <= ymin:
            ymax = ymin + 40
        y = top_y - (grip + 1) * panel_height - grip * 2
        for x, (name, _, color) in zip((left_x, right_x), OBJECTS):
            draw_panel(pdf, series[name], color, grip, x, y,
                       panel_width, panel_height, ymin, ymax)
        pdf.setFillColor(HexColor("#243746"))
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawString(left_x, y + panel_height - 7, f"Grip {grip + 1}")

        pdf.saveState()
        pdf.translate(22, y + 20 + (panel_height - 38) / 2)
        pdf.rotate(90)
        pdf.setFont("Helvetica", 8.5)
        pdf.setFillColor(HexColor("#4C5964"))
        pdf.drawCentredString(0, 0, "Final angle (degrees)")
        pdf.restoreState()

    pdf.setFont("Helvetica", 8.5)
    pdf.setFillColor(HexColor("#4C5964"))
    for x in (left_x, right_x):
        pdf.drawCentredString(x + panel_width / 2, 14, "Joint index within grip")
    pdf.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
