# USP At Risk Tracker (version 3)

An early warning system for lecturers. It reads the marks so far, compares
each student with how thousands of past students finished, and shows who to
contact first and why. It works for **any course structure**: any number of
assessments, any weights, any names, with or without a final exam.

## How to use it

1. **Add a course from its outline.** Admins and lecturers can click
   **Add a course** and upload the course outline as a PDF, Word (.docx) or Excel
   file. The system reads the course code, name, semester, campus and the
   assessment table: each assessment's name, type, weight, marked out of and due
   date. You check it on the next screen, fix anything, add or remove rows (the
   weights must add up to 100%) and set **when to warn**: the chance of failing
   at which a student is *at risk* (default 40%) and *borderline* (default 20%).
   A lecturer who adds a course is given access to it automatically.
2. **Upload this semester's marks** (admins, or the lecturer who added or
   was given the course). The course's own marksheet template can be downloaded
   from its page, with the assessments and weights filled in. A new course gets
   predictions from its first upload, because the model is trained on past
   students from every course. Because the structure comes from the outline, a marksheet whose
   headings are only the assessment names (no weights) is understood too.
3. **The results open straight away**: who is at risk, the two checks, the
   charts and every student's report. Change the thresholds at any time under
   *Thresholds and data*; the results update immediately.

Example files to try are in `static/examples/`: outlines for IS413 (PDF),
CS111 (Word), MA102 (Excel) and BIO150 (a PDF without a table), plus the
IS413 marksheet and an MA102 marksheet with plain headings.

### How the outline is read

The assessment table is the one whose header has an assessment column
(Assessment, Task, Component, Item...) and a weight column (Weight, Percentage,
Weighting, Worth, %). Total rows are skipped and a table that continues on the
next PDF page is joined. If there is no table, lines such as
`Mid Semester Test ...... 20%` are used. The course code (like IS413) and title
come from a "Course code" or "Course title" row or from the heading. Scanned
PDFs that are only images cannot be read; upload the Word version or type the
assessments in. Optionally, set `ATRISK_LLM_OUTLINE=1` and `ANTHROPIC_API_KEY`
so outlines the rules cannot read completely get a second reading by Claude
(only the outline text is sent). The outline file itself is not stored.

## The two checks

Every student goes through two checks, in this order.

**Check 1: arithmetic.** From the course weights alone: the mark already
banked, the best possible final mark and the average needed on the remaining
work. If even 100% on the rest cannot reach 50%, the student is **at risk for
certain** (chance of failing 100%). If the pass mark is already banked, they
are **on track for certain** (0%). In both cases the model is not needed.

**Check 2: random forest.** For everyone still open, a calibrated random forest
gives the chance of failing, learned from past students with known results.
Your thresholds turn that chance into *at risk*, *borderline* or *on track*.
The random forest works for any course, even one with no history of its own,
because it reads measures that every course has (average so far, standing in
the class, missed work, trend and so on). Uploading past cohorts of your own
course makes it a closer match.

## What changed in version 3

- **Trained once, stored in SQLite.** The trained model is saved inside the
  database (`data/atrisk.db`), not as a separate file, so the app loads it on
  start and never retrains on its own. (Version 2 stored full Windows paths to
  model files, so moving the project folder lost the model.) Each course's
  results are also stored, so pages open instantly.
- **Training library of real data.** The model learns from the Open University
  Learning Analytics Dataset: about 13,400 real, anonymised students in six
  modules with different assessment structures. These sit in a hidden
  *training library*, so your course list only shows the courses you add.
- **Random forest as the model,** with gradient boosting trained on the same
  folds as a comparison (Brier score 0.0855 for the random forest against
  0.0867 for gradient boosting, on courses neither had seen).
- **Checked against reality.** Trained only on 2013 to early 2014 cohorts, then
  run on the real October 2014 cohort with half the coursework marked, it caught
  69% of the students who failed, 70% of the students it flagged did fail, and
  its ranking quality (AUC) was 0.89.
- **Course setup flow** with thresholds, and results shown as soon as marks
  are uploaded.
- **Full dashboards in a cleaner layout:** predicted grades, average against
  chance of failing, class average by assessment, spread of averages, the two
  checks, and a detailed student report with charts and tables.
- **Better assistant** on every page ("Ask a question"): remembers the student
  you were discussing, answers "what if she gets 70%", drafts a supportive email.
- **Ready for Vercel** (see below). LightGBM is no longer needed.

## Run it on your computer (Windows)

1. Install **Python 3.12 or later** from python.org and tick *Add Python to PATH*.
2. Double click **run_windows.bat**. The first time it installs the packages;
   if `data/atrisk.db` is missing it builds it and trains the model (about
   3 minutes). After that it starts straight away.
3. Open **http://127.0.0.1:5000**.

Demo accounts (change these passwords under *Users* before real use):

| Account | Password | Sees |
|---|---|---|
| `admin` | `admin123` | Adds courses, sets thresholds, uploads marks, sees everything |
| `lecturer` | `lecturer123` | The courses an admin ticks for them (none at first) |

On Mac or Linux run `./run_mac_linux.sh`.

Useful commands:

```
python build_database.py            # build data/atrisk.db if it is missing parts
python build_database.py --fresh    # delete it and rebuild from data/oulad (about 3 minutes)
python build_database.py --retrain  # keep the data, train a new model version
python -m unittest discover tests   # quick checks (they work on a copy of the database)
```

## Put it on Vercel

The project deploys to Vercel as a Flask app with no extra setup
(`app.py` is the entry point; `vercel.json` and `.vercelignore` are included).

1. Put this folder in a GitHub repository, including `data/atrisk.db`
   (about 17 MB), and import it at vercel.com/new. Or install the Vercel CLI
   and run `vercel` in this folder.
2. Under *Settings, Environment Variables*, set:
   - `ATRISK_SECRET_KEY`: a long random string (keeps sign-ins secure)
   - `ATRISK_ADMIN_PASSWORD`: the admin password for the hosted copy
   - leave `ATRISK_PSEUDONYM_KEY` unset unless you rebuild the database with a
     new key (it must match the key the database was built with)
3. Deploy.

**Important limits of hosting on Vercel.** Vercel runs the app as serverless
functions with a read-only project folder. On start the app copies
`data/atrisk.db` to `/tmp` and works from that copy. So:

- Lecturers viewing results, the assistant and *Try different marks* work normally.
- Changes made on the hosted copy (uploads, new users, password changes) are
  **temporary**: they vanish when Vercel recycles the function.
- Model training is switched off there, because it takes longer than a
  serverless request may run.

The intended workflow is to do the admin work (uploads, training) on your own
computer, then redeploy with the updated `data/atrisk.db`. For a hosted copy
where changes persist, move the database to a hosted SQLite service such as
Turso, or run the app on a host with a persistent disk (a small VM, Render or
Railway with a volume).

The scikit-learn version is pinned (`scikit-learn==1.9.1`) because a saved
model can only be loaded by the same version. If you change it, retrain.

## Data

`data/oulad/` holds the Open University Learning Analytics Dataset (OULAD),
converted by `tools/prepare_oulad.py` into this system's own upload format and
loaded through the normal admin pipeline by `build_database.py`.

> Kuzilek J., Hlosta M., Zdrahal Z. *Open University Learning Analytics
> dataset.* Scientific Data 4, 170171 (2017). Licence CC BY 4.0.
> https://analyse.kmi.open.ac.uk/open-dataset

How it was mapped:

- Course = module (AAA to FFF). GGG is left out because none of its coursework carries weight.
- Assessments: tutor-marked assignments (typed Coursework) and computer-marked
  quizzes (typed Quiz) with a weight above zero, in date order, scaled to 50% of
  the course, plus a Final Exam worth 50%.
- Missing coursework counts as 0 (not submitted). Exam scores are only
  published for some modules; elsewhere the exam is blank.
- Outcome: Pass or Distinction = Pass, Fail = Fail. Students who withdrew are
  left out of the training data.
- History = 2013B, 2013J, 2014B, stored as the training library. The 2014J
  presentation, cut where about half the coursework had been marked, is scored
  once during the build and compared with its real outcomes
  (`data/oulad/actual_outcomes/`) for the reality check on the *How predictions
  work* page; it is not kept as a course.

Because the outcomes are Pass or Fail only, the chance of failing is trained,
but letter grades are estimated from the current average. Upload history with
letter grades or final marks to train grade predictions too.

To use your own data, sign in as admin and click **Add a training set**
(Admin home or the upload page). Give the set a name (60 characters or fewer),
optionally a cohort or year, and choose a .csv or .xlsx of past results with
final grades. The set is stored in the training library under a code starting
`TRAIN-`, which no real course uses. Saving retrains the model in the
background (a few minutes); the Admin home shows when the set is in the active
model, and from then on predictions for every course use it. Adding another
file under the same name adds it to that set as another cohort. Lecturers then
add their courses and upload this semester's marks; no history of their own is
needed.

Past results are never rejected because of their layout; each repair is shown
on the check screen before saving:

- The student ID column is found by name (Student, Student code, ...), then
  as a column of unique labels; with neither, students are numbered by row.
- A column that is the total of 2 to 6 other assessment columns (such as
  *Continuous Assessment (60%)*) is set aside so marks are not counted twice,
  and its weight is shared between its parts in proportion to their highest
  marks.
- Assessments with no weight share what is left equally, and weights that do
  not add up to 100% are scaled to 100%.
- With no maximum mark, the highest mark in the file (rounded up) is used.
- `-`, NS, ABS, text and out of range marks count as missing; repeated IDs
  keep the first row.
- `D*` is a fail (D). Other grades off the scale (such as EX) use the grade
  for the final mark; students with no usable outcome are left out.

This semester's marksheets keep the strict checks, because their results are
shown to lecturers.

## Roles

| Role | Can do |
|---|---|
| **Administrator** | Add named training sets (past results) for the global model, create courses, upload and map marks, replace or roll back datasets, assign lecturers, retrain or roll back the model, view the audit log. |
| **Lecturer** | Add courses, upload this semester's marks for their own courses, and see the results: dashboard, student reports, what-if analysis, export, assistant. |

## Status rules

A student is **At risk** if a pass is no longer possible, the chance of failing
is above the course's risk threshold (40% by default), or a fail is the likely
outcome. **Borderline** covers a moderate chance of failing (20% by default), a
very high required average, falling marks or a missed submission. Admins set
the thresholds per course on its *Manage data* page.

## Privacy

Each student gets a keyed code (`P` + 10 characters), different per course,
which the system uses internally and in page addresses. For this semester's
marks the student ID from the marksheet is also stored and is what lecturers
see on the dashboard, student pages, export and assistant; search accepts it
too. Past results (training sets) keep the code only. Marks uploaded before
IDs were kept show the code until they are uploaded again. Names and other
personal columns are never stored.

Because real IDs are now in `data/atrisk.db`, treat the database file as
personal data: keep it out of public repositories and deploy it only where
the course's staff may see the IDs.

The optional AI wording for the assistant (`ATRISK_LLM_ASSISTANT=1` with
`ANTHROPIC_API_KEY`) sends only the question and the computed answer with
pseudonymous codes; the browser swaps each code for the student ID when it
shows the answer. It is off by default.

## How the prediction works

1. **Feasibility.** From the weights: mark banked, best possible final mark, and
   the average needed on the remaining work for each grade.
2. **Universal features.** Every student is described the same way whatever the
   course: weighted average so far, share of the course assessed, gap to the
   class (also as a z score), lowest and highest mark, spread, share of weight
   below 50%, trend, missed submissions, averages by type of assessment, and
   course context.
3. **One global model** trained on snapshots of past students at several stages
   of their course, so it can predict from any point in the semester.
4. **Calibration** (isotonic regression), so "30% chance" means about 30 in 100
   similar past students failed.
5. **Honest evaluation.** Cross validation holds out whole courses and whole
   cohorts, plus the out-of-time reality check above.
6. **Explanations.** Each prediction lists the measures that moved it most,
   found by resetting each measure to a typical value and seeing how much the
   chance of failing changes.
7. **Cold start.** A course with no history gets a clearly labelled simulation instead of a trained prediction.

## Project files

| File | Purpose |
|---|---|
| app.py | Web routes, roles, admin portal |
| database.py | SQLite: courses, datasets, model versions (model stored as a BLOB), summary cache, audit log |
| build_database.py | Builds `data/atrisk.db` from `data/oulad`, trains once, precomputes dashboards, runs the reality check |
| ml_engine.py | Training, algorithm choice, calibration, cross validation, explanations, simulation, drift |
| analytics.py | Feasibility, status, concerns, recommendations, cached course summaries |
| assistant.py, static/assistant.js | The assistant and its chat panel |
| features.py | Structure independent features |
| schema_mapping.py, ingest.py | Reading any file layout, mapping, validation, saving |
| outline.py | Reads course outlines (PDF, Word, Excel): course details and the assessment table |
| tools/prepare_oulad.py | Converts the raw OULAD files into upload files |
| vercel.json, .vercelignore, .python-version | Vercel deployment |
| tests/test_core.py | Quick checks |
