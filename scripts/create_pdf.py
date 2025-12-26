#!/usr/bin/env python
"""Generate PDF with conceptual diagrams for presentation."""

from fpdf import FPDF
import os

# Paths to generated images
IMAGE_DIR = r"C:\Users\User\.gemini\antigravity\brain\2382d71f-e86c-44b2-b7a1-dc9137c227bf"
OUTPUT_DIR = r"d:\multimodal_rl_research\experiments\figures"

# Find the images
images = []
for f in os.listdir(IMAGE_DIR):
    if f.endswith('.png'):
        images.append(os.path.join(IMAGE_DIR, f))

print(f"Found {len(images)} images")
for img in images:
    print(f"  - {os.path.basename(img)}")

# Create PDF
class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Compositional Skill Learning in Multimodal RL', 0, 1, 'C')
        self.set_font('Helvetica', 'I', 10)
        self.cell(0, 5, 'Conceptual Diagrams for Presentation', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Title page
pdf.add_page()
pdf.set_font('Helvetica', 'B', 24)
pdf.ln(40)
pdf.cell(0, 20, 'Multimodal Compositional RL', 0, 1, 'C')
pdf.set_font('Helvetica', '', 16)
pdf.cell(0, 10, 'Research Presentation Materials', 0, 1, 'C')
pdf.ln(20)
pdf.set_font('Helvetica', 'I', 12)
pdf.cell(0, 10, 'Extending "From f(x) and g(x) to f(g(x))"', 0, 1, 'C')
pdf.cell(0, 10, 'to Vision-Language Models', 0, 1, 'C')

# Add each image on a new page
captions = {
    'skill_composition': 'Figure 1: Skill Composition Framework\n\nShows how atomic skills f(x) (vision) and g(x) (language) combine through RL training into composed skill f(g(x)) for VQA.',
    'rl_vs_supervised': 'Figure 2: RL vs Supervised Learning\n\nCompares training signals: supervised uses cross-entropy loss on labels, RL uses reward signal based on answer correctness.',
    'architecture': 'Figure 3: System Architecture\n\nFrozen CLIP vision encoder connects through trainable projection to T5 language model, with RL reward feedback loop.',
}

for img_path in sorted(images):
    pdf.add_page()
    
    # Find caption
    basename = os.path.basename(img_path)
    caption = None
    for key, cap in captions.items():
        if key in basename:
            caption = cap
            break
    
    if caption is None:
        caption = f"Figure: {basename}"
    
    # Add image
    try:
        pdf.image(img_path, x=10, y=40, w=190)
    except Exception as e:
        pdf.set_font('Helvetica', '', 12)
        pdf.cell(0, 20, f"Image: {basename}", 0, 1, 'C')
        print(f"Warning: Could not add image {basename}: {e}")
    
    # Add caption
    pdf.set_y(200)
    pdf.set_font('Helvetica', '', 11)
    pdf.multi_cell(0, 6, caption)

# Also add experiment figures if available
exp_figures_dir = os.path.join(OUTPUT_DIR)
if os.path.exists(exp_figures_dir):
    exp_captions = {
        'fig1_method_comparison': 'Figure 4: Method Comparison\n\nCompares accuracy across Frozen (no training), Supervised, and RL methods.',
        'fig2_learning_rate': 'Figure 5: Learning Rate Effect\n\nShows how accuracy varies with different learning rates. Optimal around 2e-4.',
        'fig3_reward_functions': 'Figure 6: Reward Function Comparison\n\nCompares different RL reward functions: exact match, partial match, progressive, etc.',
        'fig4_question_types': 'Figure 7: Performance by Question Type\n\nBreaks down accuracy by color, shape, count, and spatial questions.',
        'fig5_experiment_summary': 'Figure 8: Experiment Summary\n\nTop 20 experiments ranked by accuracy.',
    }
    
    for f in sorted(os.listdir(exp_figures_dir)):
        if f.endswith('.png'):
            img_path = os.path.join(exp_figures_dir, f)
            pdf.add_page()
            
            # Find caption
            caption = None
            for key, cap in exp_captions.items():
                if key in f:
                    caption = cap
                    break
            
            if caption is None:
                caption = f"Figure: {f}"
            
            try:
                pdf.image(img_path, x=10, y=40, w=190)
            except Exception as e:
                pdf.set_font('Helvetica', '', 12)
                pdf.cell(0, 20, f"Image: {f}", 0, 1, 'C')
            
            pdf.set_y(200)
            pdf.set_font('Helvetica', '', 11)
            pdf.multi_cell(0, 6, caption)

# Save PDF
output_path = os.path.join(OUTPUT_DIR, 'presentation_figures.pdf')
pdf.output(output_path)
print(f"\n✅ PDF created: {output_path}")
