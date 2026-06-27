# Publishing Checklist

## Before Push

1. Review `README.md` and `LESSONS_LEARNED.md` for any personal wording you want to adjust.
2. Confirm no model weights were copied into `birdclef/models/`.
3. Confirm `birdclef/output/` contains only `.gitkeep`.
4. Confirm no competition data folder is present.

## Suggested Repo Naming

Good options:

- `birdclef-2026-archive`
- `birdclef-2026-lessons-learned`
- `cibuco-boriken-birdclef-2026-archive`

## Local Git Commands

```powershell
cd D:\Manatuabon\staging\birdclef-2026-archive
git init
git add .
git commit -m "archive: preserve BirdCLEF 2026 project and lessons learned"
git branch -M main
git remote add origin <your-new-repo-url>
git push -u origin main
```

## Recommended GitHub Description

Historical archive of the BirdCLEF 2026 competition project, preserved as an engineering record of the system design, experiments, and lessons learned.

## Recommended Topics

- `birdclef`
- `bioacoustics`
- `kaggle`
- `audio-classification`
- `machine-learning`
- `signal-processing`
- `wildlife-monitoring`
