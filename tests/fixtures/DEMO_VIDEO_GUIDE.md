# Demo Video Recording Guide

## Purpose
This guide helps you record high-quality demo videos for testing and showcasing the AI PT Assistant. Good demo videos ensure consistent testing and impressive presentations.

---

## Recording Setup Requirements

### Camera Placement
- **Distance:** 6-10 feet (2-3 meters) from exercise area
- **Height:** Hip height for most standing exercises
- **Angle:** Perpendicular to movement plane (side view for most exercises)
- **Stability:** Use tripod or stable surface (no handheld)

```
        [Camera]
            |
            |  6-10 ft
            |
    ----------------
    |   Exercise   |
    |     Area     |
    ----------------
```

### Lighting
- **Type:** Even, diffused lighting (avoid harsh shadows)
- **Direction:** Face light source (window or lamp in front, not behind)
- **Avoid:** Backlighting, direct sunlight, flickering lights

### Background
- **Ideal:** Plain wall, solid color
- **Acceptable:** Minimal clutter, no mirrors
- **Avoid:** Busy patterns, other people, reflective surfaces

### Clothing
- **Best:** Fitted athletic wear in solid colors
- **Acceptable:** Contrasting color to background
- **Avoid:** Baggy clothes, patterns, colors matching background

### Resolution & Framerate
- **Minimum:** 720p (1280x720)
- **Recommended:** 1080p (1920x1080)
- **Framerate:** 30fps minimum (60fps ideal for fast movements)

---

## Demo Script: Knee Extension

### Setup
1. Sit on chair/bench with back supported
2. Ensure full body visible from side
3. Start with knee bent at ~90°

### Recording (Good Form - 10 reps)
```
Rep 1-3:  Perfect form
          - Back against chair
          - Full extension to ~170°
          - Controlled 2-second up, 2-second down

Rep 4-6:  Perfect form
          - Maintain consistent tempo
          - Smooth movement

Rep 7-10: Perfect form (showing consistency)
```

### Recording (Intentional Form Errors - 3 reps)
```
Error Rep 1: Hip lifting off seat
             - Lean back slightly as leg extends
             - Visible hip rise

Error Rep 2: Using momentum
             - Quick, jerky movement
             - Swing leg up instead of controlled lift

Error Rep 3: Incomplete range of motion
             - Only extend to ~120° instead of full 170°
```

### Total Duration: ~2-3 minutes

---

## Demo Script: Shoulder Flexion

### Setup
1. Stand with side to camera
2. Arms at sides, relaxed
3. Feet shoulder-width apart

### Recording (Good Form - 8 reps)
```
Rep 1-4:  Perfect form
          - Raise arm smoothly to 170°
          - Keep elbow slightly bent
          - Core engaged, no back arch

Rep 5-8:  Perfect form with varying ROM
          - Rep 5: Full range (170°)
          - Rep 6: Slight reduction (160°)
          - Rep 7-8: Back to full range
```

### Recording (Intentional Form Errors - 2 reps)
```
Error Rep 1: Back arching
             - Lean back as arm rises
             - Visible trunk extension

Error Rep 2: Shoulder shrugging
             - Raise shoulder toward ear
             - Obvious trap engagement
```

### Total Duration: ~2 minutes

---

## Demo Script: Squats

### Setup
1. Stand facing camera (or 45° angle)
2. Feet shoulder-width apart
3. Arms can be extended forward for balance

### Recording (Good Form - 5 reps)
```
Rep 1-3:  Perfect form
          - Knees track over toes
          - Chest up, back straight
          - Depth to ~90° knee angle
          - Weight in heels

Rep 4-5:  Perfect form (showing consistency)
```

### Recording (Form Errors - 5 reps)
```
Error Rep 1: Knee valgus (knees caving in)
             - Let knees collapse inward at bottom
             - Visible knock-knee position

Error Rep 2: Forward lean
             - Excessive trunk lean
             - Chest drops toward knees

Error Rep 3: Heel rise
             - Come onto toes at bottom
             - Visible heel lift

Error Rep 4: Asymmetric depth
             - One side goes deeper than other
             - Visible hip shift

Error Rep 5: Partial rep
             - Only go to ~45° instead of 90°
```

### Total Duration: ~2-3 minutes

---

## Edge Case Videos

Record these to test system robustness:

### 1. Partial Visibility
- **Setup:** Position so legs are cut off at knee
- **Purpose:** Test handling of missing keypoints
- **Duration:** 30 seconds of movement

### 2. Multiple People
- **Setup:** Have second person walk through frame
- **Purpose:** Test person detection/tracking
- **Duration:** 1 minute with occasional interference

### 3. Poor Lighting
- **Setup:** Dim lights significantly
- **Purpose:** Test low-light detection
- **Duration:** 30 seconds

### 4. Fast Movement
- **Setup:** Perform exercise at 2x normal speed
- **Purpose:** Test temporal smoothing
- **Duration:** 30 seconds

### 5. Slow Movement
- **Setup:** Perform exercise very slowly (5+ seconds per phase)
- **Purpose:** Test state machine with slow transitions
- **Duration:** 1 minute

### 6. Camera Angle Variations
- **Setup:** Record from front, 45°, and side angles
- **Purpose:** Test angle calculation from different views
- **Duration:** 30 seconds each angle

---

## File Naming Convention

Use consistent naming for organization:

```
{exercise}_{condition}_{date}.mp4

Examples:
- knee_extension_good_form_20240115.mp4
- knee_extension_bad_form_20240115.mp4
- squat_knee_valgus_20240115.mp4
- shoulder_flexion_edge_case_low_light_20240115.mp4
```

---

## Quality Checklist

Before using a demo video, verify:

- [ ] Full body visible in frame throughout
- [ ] Stable camera (no shake)
- [ ] Adequate lighting (face visible)
- [ ] Clean background
- [ ] Audio not required (can be silent)
- [ ] Exercise performed at realistic pace
- [ ] Both good and bad form represented
- [ ] File plays correctly in video player

---

## Sample Video Inventory

Recommended demo video set:

| File | Exercise | Duration | Purpose |
|------|----------|----------|---------|
| knee_ext_demo.mp4 | Knee Extension | 3 min | Main demo |
| knee_ext_errors.mp4 | Knee Extension | 1 min | Error detection |
| shoulder_flex_demo.mp4 | Shoulder Flexion | 2 min | Main demo |
| squat_demo.mp4 | Squat | 3 min | Main demo |
| squat_valgus.mp4 | Squat | 1 min | Knee valgus detection |
| edge_low_light.mp4 | Any | 30 sec | Low light test |
| edge_partial.mp4 | Any | 30 sec | Partial visibility |
| edge_multi_person.mp4 | Any | 1 min | Multi-person test |

---

## Tips for Great Demos

1. **Practice first** - Know the exercise before recording
2. **Use countdown** - Start recording, count down, then begin
3. **Exaggerate errors** - Make form issues obvious for detection
4. **Keep consistent** - Same setup for comparison videos
5. **Review immediately** - Check recording before moving on
6. **Multiple takes** - Better to have options
7. **Note timestamps** - Record when errors occur for testing

---

## Using Videos for Testing

```python
# Quick test with a demo video
python analyze.py video \
    --input videos/knee_ext_demo.mp4 \
    --exercise knee_extension \
    --output output_results/knee_ext_analyzed.mp4

# Batch test all videos
for video in videos/*.mp4; do
    python analyze.py video --input "$video" --exercise knee_extension
done
```

---

## Troubleshooting Common Issues

### Problem: Pose not detected
- **Cause:** Poor lighting or background
- **Fix:** Improve lighting, use plain background

### Problem: Jerky angle readings
- **Cause:** Fast movement or low framerate
- **Fix:** Slow down movement, increase framerate

### Problem: Wrong person tracked
- **Cause:** Multiple people in frame
- **Fix:** Clear the frame, record alone

### Problem: Inconsistent rep counting
- **Cause:** Angle thresholds not matching movement
- **Fix:** Adjust thresholds in config.yaml or exaggerate ROM
