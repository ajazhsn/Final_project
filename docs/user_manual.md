# User Manual
## Plant Disease Detection System
### For Non-Technical Users

---

## Welcome!

This application helps farmers and agricultural workers identify plant diseases from leaf photographs. Simply upload a photo of a plant leaf and the system will tell you what disease (if any) is present and what to do about it.

---

## Getting Started

### What You Need
- A web browser (Chrome, Firefox, or Edge)
- A photo of a plant leaf (JPG or PNG format)
- The application must be running (ask your administrator)

### Opening the Application
1. Open your web browser
2. Type this address: `http://localhost:8501`
3. Press Enter
4. The Plant Disease Detector will open

---

## Using the Application

### Checking the System is Ready
Look at the top-left corner of the screen. You should see:
- ✅ **API Connected** — green box means the system is ready
- ❌ **API Offline** — red box means the system is not running (contact your administrator)

---

### Single Image Detection (Most Common Use)

**Step 1:** Click **"🔍 Single Detection"** in the left menu

**Step 2:** Click **"Browse files"** or drag and drop your leaf photo

**Step 3:** Wait 1-2 seconds for the analysis

**Step 4:** Read your results:

🔴 Tomato_Late_blight
Confidence: 99.3%
Severity: High
Treatment: Apply fungicide immediately. Remove and
destroy infected plants. Improve air circulation.

**Understanding the results:**
- 🟢 **Green circle** = Healthy plant, no action needed
- 🟡 **Yellow circle** = Medium severity disease, treatment recommended
- 🔴 **Red circle** = High severity disease, act immediately
- **Confidence %** = How sure the system is (above 80% is reliable)

---

### Bulk Detection (Multiple Images)

Use this when you have many leaf photos to check at once.

**Step 1:** Click **"📦 Bulk Detection"** in the left menu

**Step 2:** Choose upload type:
- **Multiple Images** — select several photos at once
- **ZIP File** — upload a ZIP file containing many photos

**Step 3:** Click **"🚀 Run Batch Analysis"**

**Step 4:** Review results — each image shows its diagnosis

---

### Supported Plants and Diseases

| Plant | Disease | Severity |
|-------|---------|----------|
| 🥔 Potato | Early Blight | Medium |
| 🥔 Potato | Late Blight | High |
| 🥔 Potato | Healthy | None |
| 🍅 Tomato | Early Blight | Medium |
| 🍅 Tomato | Late Blight | High |
| 🍅 Tomato | Healthy | None |

---

### Treatment Guide

**Potato Early Blight**
Apply fungicide containing chlorothalonil. Remove infected leaves. Ensure proper plant spacing for air circulation.

**Potato Late Blight**
Apply copper-based fungicide immediately. Destroy infected plants to prevent spread. Avoid overhead irrigation.

**Tomato Early Blight**
Use fungicide spray on affected areas. Remove lower infected leaves. Apply mulch around the base of the plant.

**Tomato Late Blight**
Apply fungicide immediately. Remove and destroy infected plants. Improve air circulation around remaining plants.

---

## Tips for Best Results

✅ **Good photo tips:**
- Take photos in good natural lighting
- Make sure the leaf fills most of the photo
- Keep the camera steady — avoid blurry photos
- Use the diseased side of the leaf

❌ **Avoid:**
- Very dark or very bright photos
- Blurry or out-of-focus images
- Photos with multiple leaves overlapping
- Photos taken at night

---

## Frequently Asked Questions

**Q: The confidence is only 60% — should I trust the result?**
A: Confidence below 80% means the system is unsure. Take another photo in better lighting or consult an agricultural expert.

**Q: My plant is not potato or tomato — can I still use it?**
A: Currently the system only supports potato and tomato leaves. More plants will be added in future updates.

**Q: How long does analysis take?**
A: Single image: 1-2 seconds. Bulk analysis: depends on number of images, typically 2-5 seconds per image.

**Q: Can I use this on my phone?**
A: Yes — open the same address in your phone's browser.

---

## Getting Help

If the application is not working:
1. Check that **✅ API Connected** is showing in the top left
2. Try refreshing the page (press F5)
3. Contact your system administrator

---

*Plant Disease Detection System — DA5402 MLOps Project*