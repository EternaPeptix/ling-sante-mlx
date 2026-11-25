"""Generate a 600-sequence strictly disjoint LingSante-3.1-flash self-distillation dataset for EAGLE-3 v3.

Stages:
  --stage check     : Run anti-leakage guard and verify 600 distinct train prompts (0 overlaps).
  --stage rollouts  : Generate C=4 greedy rollouts (max 256 tokens) with BatchGenerator -> rollouts.json.
  --stage capture   : Prefill full sequences with target model, extract layer (1,18,35,52,70) features,
                      inline h_final, and top-32 teacher logits -> sharded safetensors.
  --stage all       : Run check -> rollouts -> capture.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_root / "vendor"), str(_root)]

from tools.gen_dflash_data import FORBIDDEN_SUBSTRINGS, TARGET_LAYER_IDS
from tools.gen_dflash_data import _build_prompts as _orig_prompts

EXTRA_FORBIDDEN = [
    "potassium permanganate",
    "hydrochloric acid",
    "chlorine gas",
    "tert-butyl bromide",
    "sodium methoxide",
    "daughter cells",
    "chromosome number",
    "crossing over",
    "right coronary",
    "stage 3b chronic kidney",
]


def _build_prompts() -> list[tuple[int, str, str, str]]:
    """Build 600 diverse train prompts across med (140), chem (80), bio (80), math (100), code (120), tech (80)."""
    med = [
        # Cardiology & Vascular (16)
        "A 74-year-old man with hypertension and hyperlipidemia reports intermittent jaw tightness when walking briskly uphill, resolving within four minutes of rest. Outline stress imaging modalities stratified by resting ECG interpretability, Duke treadmill scoring, and antianginal escalation.",
        "Explain the hemodynamics of mitral stenosis versus chronic mitral regurgitation, characteristic diastolic rumble versus holosystolic murmur radiation, left atrial remodeling, and percutaneous valvuloplasty Wilkin criteria.",
        "A 61-year-old woman presents three weeks post- anterior wall infarction with pleuritic precordial pain, fever 38.1 C, ESR 85 mm/hr, and a small pericardial effusion. Compare Dressler syndrome versus post-MI free wall rupture and papillary muscle avulsion.",
        "Detail diagnostic electrophysiology criteria for Wolff-Parkinson-White preexcitation (delta wave, short PR), orthodromic versus antidromic AVRT management, and why AV nodal blockers are contraindicated during preexcited atrial fibrillation.",
        "A 53-year-old man develops recurrent exertional lightheadedness and a systolic crescendo-decrescendo murmur that intensifies with Valsalva and standing while softening with squatting. Explain sarcomere mutations, ICD sudden death predictors, and alcohol septal ablation versus myectomy.",
        "Compare Brugada type 1 coved ST elevation, long QT syndrome types 1-3 triggers and KCNQ1/KCNH2/SCN5A channelopathies, and catecholaminergic polymorphic ventricular tachycardia management.",
        "A 67-year-old woman with obesity and diabetes presents with dyspnea on exertion, normal LVEF 62%, E/e' ratio 18, and concentric LV remodeling. Detail heart failure with preserved ejection fraction pathophysiology, diagnostic H2FPEF scoring, and SGLT2 inhibitor/MRA trial evidence.",
        "Explain chronic thromboembolic pulmonary hypertension diagnosis via V/Q scan versus CTPA, right ventricular coupling, pulmonary endarterectomy candidacy, and riociguat cGMP pharmacology.",
        "A 79-year-old man has episodic unprovoked syncope, complete AV dissociation on telemetry with ventricular escape rate 34 bpm, and narrow QRS complexes. Distinguish nodal versus infranodal conduction blocks and permanent dual-chamber pacemaker pacing modes.",
        "Compare dilated, arrhythmogenic right ventricular (desmosomal epsilon wave), and Takayasu arteritis vascular pathologies across diagnostic biopsy/imaging findings and guideline therapeutics.",
        "A 44-year-old smoker presents with asymmetric cold pulseless left upper extremity, subclavian bruit, and vertigo during left arm exercise. Explain subclavian steal siphon physiology and carotid-subclavian bypass indications.",
        "Contrast cardiogenic shock versus distributive septic shock versus obstructive tamponade across Swan-Ganz pulmonary artery catheter hemodynamic profiles (CVP, PCWP, CI, SVR, SvO2).",
        "A 60-year-old woman receiving trastuzumab and prior doxorubicin develops asymptomatic LVEF decline from 64% to 44%. Compare Type I anthracycline irreversible myocyte necrosis against Type II HER2 reversible dysfunction.",
        "Explain carotid endarterectomy versus transfemoral stenting selection based on NASCET symptomatic 70-99% stenosis thresholds, perioperative dual antiplatelet protocols, and cranial nerve injury anatomy.",
        "Compare aneurysmal degeneration of abdominal aorta growth surveillance intervals, Laplace wall stress rupture physics, open surgical repair versus endovascular EVAR endoleak Types I-IV.",
        "A 50-year-old marathon runner develops chest tightness and inverted precordial T waves after acute viral gastroenteritis with troponin I 4.2 ng/mL and unobstructed coronaries. Outline cardiac MRI Lake Louise criteria and exercise restriction durations.",
        # Pulmonology & Critical Care (14)
        "Explain alveolar-arterial oxygen gradient equation derivation, five pathophysiologic mechanisms of hypoxemia (hypoventilation, V/Q mismatch, shunt, diffusion impairment, low FiO2), and response to 100% supplemental O2.",
        "A 39-year-old nonsmoking woman presents with progressive dyspnea, basilar Velcro crackles, anti-Jo-1 antibodies, mechanic's hands, and honeycombing sparing costophrenic angles. Differentiate NSIP from UIP on HRCT and biopsy.",
        "Compare cystic fibrosis CFTR mutation classes I-VI (gating vs trafficking defects), triple-combination elexacaftor-tezacaftor-ivacaftor mechanism, Burkholderia cepacia prognosis, and pancreatic enzyme titration.",
        "A 57-year-old immunocompromised recipient of allogeneic stem cell transplant presents with pleuritic pain, hemoptysis, galactomannan positivity, and halo/air-crescent nodules on chest CT. Outline angioinvasive aspergillosis diagnosis and voriconazole/isavuconazole pharmacology.",
        "Explain polysomnography AHI severity grading for obstructive versus central Cheyne-Stokes sleep apnea, genioglossus airway collapsibility, CPAP auto-titration, and cardiometabolic sequelae.",
        "A 62-year-old shipbuilder presents with unilateral pleural thickening encasing the right lung and hemorrhagic exudative effusion negative on three cytology taps. Compare mesothelioma versus benign asbestos plaques and talc pleurodesis.",
        "Detail neuromuscular respiratory failure monitoring using vital capacity (<15-20 mL/kg) and negative inspiratory force (-30 cmH2O), early intubation vs noninvasive support pitfalls, and bulbar aspiration prevention.",
        "Contrast hypersensitivity pneumonitis (acute neutrophilic vs chronic fibrotic centrilobular ground-glass nodules), sarcoidosis noncaseating granulomas, and silicosis eggshell nodal calcifications.",
        "A 26-year-old tall thin man feels sudden right chest pain and dyspnea at rest, showing hyperlucent hemithorax without lung markings and tracheal deviation to the left. Outline tension physiology, needle decompression 4th/5th intercostal space, and VATS pleurectomy indications.",
        "Explain bronchial carcinoid tumor versus small-cell carcinoma versus peripheral adenocarcinoma immunohistochemistry (TTF-1, napsin A, chromogranin) and actionable EGFR/ALK/ROS1 biomarker testing.",
        "A 48-year-old farmer has recurrent wheezing, brownish mucus plugs, central bronchiectasis on CT, total IgE >1,000 IU/mL, and Aspergillus skin reactivity. Explain allergic bronchopulmonary aspergillosis staging and corticosteroid-itraconazole synergy.",
        "Detail hemodynamic effects of excessive PEEP on right ventricular afterload, West lung zones 1-3 transmural pressures, auto-PEEP breath stacking waveforms, and permissive hypercapnia safety bounds.",
        "Compare diffuse alveolar hemorrhage syndromes (Goodpasture anti-GBM linear IF, lupus capillaritis, idiopathic hemosiderosis) using bronchoalveolar lavage serial aliquots and rising DLCO artifact.",
        "A 70-year-old hospitalized patient develops sudden agitation, tachypnea, and hypoxemia 24 hours after femur intramedullary nailing, alongside axillary petechiae. Explain fat embolism syndrome biochemical free-fatty-acid theory and Gurd criteria.",
        # Nephrology & Fluid/Electrolytes (14)
        "Explain renal autoregulation via myogenic arteriolar reflex and tubuloglomerular feedback at the macula densa, and predict GFR/RPF/filtration fraction changes under afferent dilation versus efferent constriction.",
        "A 28-year-old man presents with gross hematuria two days after pharyngitis, BP 146/92 mmHg, and normal C3/C4 levels. Contrast IgA nephropathy timing against post-streptococcal glomerulonephritis subepithelial humps.",
        "Compare membranous nephropathy (PLA2R antibody subepithelial spikes), focal segmental glomerulosclerosis (APOL1 risk alleles), and membranoproliferative glomerulonephritis (C3 nephritic factor) histology and outcomes.",
        "A 37-year-old patient taking lithium for bipolar disorder presents with nocturia, polydipsia, serum Na 151 mEq/L, and urine osmolality 165 mOsm/kg unresponsive to exogenous DDAVP. Explain ENaC principal-cell lithium accumulation and amiloride plus thiazide paradox.",
        "Contrast distal Type 1, proximal Type 2 (Fanconi syndrome glycosuria/phosphaturia), and Type 4 hyperkalemic renal tubular acidosis using urine anion gap, urine pH after acid load, and nephrolithiasis tendency.",
        "Explain peritoneal dialysis adequacy (Kt/V), high versus low peritoneal equilibration transporter membrane physiology, exit-site versus cloudy-effluent peritonitis culture criteria, and intraperitoneal vancomycin/ceftazidime dosing.",
        "A 63-year-old woman develops livedo reticularis, purple toes, eosinophilia, and acute kidney injury three weeks after coronary angiography. Detail cholesterol crystal atheroembolic pathology versus contrast-associated AKI recovery trajectories.",
        "Contrast uric acid, calcium oxalate, cystine (hexagonal crystals, cyanide-nitroprusside screen), and magnesium ammonium phosphate staghorn calculi pathogenesis, metabolic 24-hour urine evaluation, and potassium citrate alkalization.",
        "Explain Bartter syndrome (NKCC2) versus Gitelman syndrome (NCC) versus Liddle syndrome (constitutive ENaC) serum potassium, urinary calcium excretion, blood pressure, and renin-aldosterone profiles.",
        "A 45-year-old transplant recipient on tacrolimus, mycophenolate, and prednisone has creatinine rising from 1.2 to 2.1 mg/dL four months post-graft. Contrast calcineurin afferent arteriolar vasoconstriction, polyoma BK nephropathy decoy cells, and T-cell-mediated vs antibody-mediated rejection.",
        "Derive free-water clearance and electrolyte-free water clearance equations, applying them to predict whether normal saline infusions worsen hyponatremia in SIADH (desalination phenomenon).",
        "Compare hemolytic uremic syndrome Shiga-toxin endothelial Gb3 binding, atypical complement Factor H-mediated TMA (eculizumab C5 blockade), and scleroderma renal crisis malignant hypertension.",
        "A 55-year-old marathoner collapses after drinking 6 L of plain water during a humid 4-hour race, presenting with seizures and serum Na 118 mEq/L. Outline acute symptomatic hypertonic 3% saline 100 mL bolus protocol.",
        "Explain renal phosphate handling by NaPi-IIa/c cotransporters, FGF23-Klotho suppression of 1-alpha-hydroxylase, and X-linked hypophosphatemic rickets targeted burosumab blockade.",
        # Neurology & Psychiatry (14)
        "Compare localization patterns of cranial nerves III (pupil-sparing microvascular vs compressive PComm aneurysm), IV (extorsion/head tilt), VI (abduction/elevated ICP false localizer), and cavernous sinus thrombosis.",
        "A 66-year-old man develops sudden vertigo, dysphagia, hoarseness, ipsilateral facial pain/temperature loss, Horner syndrome, and contralateral body pinprick loss. Map every deficit on a cross-section of the dorsolateral medulla.",
        "Contrast vascular dementia, dementia with Lewy bodies (REM behavior disorder, neuroleptic hypersensitivity, alpha-synuclein), frontotemporal degeneration (Pick bodies vs TDP-43), and normal pressure hydrocephalus gait apraxia.",
        "Explain trigeminal neuralgia microvascular arterial compression physiology, carbamazepine voltage-gated Na+ channel frequency dependence, versus glossopharyngeal neuralgia and cluster headache hypothalamic periodicity.",
        "A 23-year-old woman with obesity presents with pulsatile headache worse on waking, transient visual obscurations, bilateral abducens palsy, and normal MRI brain parenchyma with empty sella. Detail idiopathic intracranial hypertension LP opening pressure criteria, acetazolamide CSF carbonic anhydrase blockade, and optic nerve sheath fenestration.",
        "Compare Huntington disease CAG repeat striatal GABAergic medium spiny neuron loss, Wilson disease ATP7B ceruloplasmin/Kayser-Fleischer chelation (trientine/zinc), and Friedreich ataxia GAA frataxin.",
        "Detail acute bipolar mania vs depressive episode pharmacology: lithium narrow therapeutic index (0.6-1.2 mEq/L) thyroid/renal toxicity, valproate teratogenicity, lamotrigine titration Stevens-Johnson risk, and metabolic monitoring for second-generation antipsychotics.",
        "Contrast transient global amnesia, psychogenic nonepileptic spells, absence seizures (3 Hz spike-and-wave T-type calcium channel ethosuximide), and mesial temporal lobe focal impaired-awareness automatisms.",
        "A 71-year-old man has rapidly progressive cognitive decline over 10 weeks with startle myoclonus, cerebellar ataxia, 1 Hz periodic sharp-wave EEG complexes, and cortical ribboning on DWI MRI. Explain prion PrPSc misfolding propagation and RT-QuIC CSF sensitivity.",
        "Explain amyotrophic lateral sclerosis upper and lower motor neuron split-hand pathology, fasciculations without sensory loss, Riluzole glutamate modulation, Edaravone, and diaphragmatic pacing.",
        "Compare obsessive-compulsive disorder exposure-response prevention CBT plus high-dose SSRI therapy against generalized anxiety disorder, PTSD trauma-focused desensitization, and panic disorder interoceptive exposure.",
        "A 47-year-old man experiences brief episodes of spinning triggered only when rolling onto his right side in bed, showing upbeat-torsional nystagmus after a 4-second latency that fatigues on repeat testing. Explain posterior canalolithiasis mechanics.",
        "Contrast cerebral venous sinus thrombosis risk factors (prothrombin G20210A, OCP, mastoiditis), delta sign on contrast CT/MRV, and why systemic anticoagulation remains safe despite venous hemorrhagic transformation.",
        "Explain spinal cord herniation/hemisection (Brown-Sequard ipsilateral corticospinal/dorsal column vs contralateral 1-2 levels below spinothalamic loss), central cord syndrome hyperextension mechanics, and anterior spinal artery infarction.",
        # Endocrinology & Reproductive (14)
        "Compare monogenic MODY diabetes subtypes (GCK mild fasting hyperglycemia vs HNF1A sulfonylurea hypersensitivity), latent autoimmune diabetes in adults (LADA), and type 2 beta-cell amyloid deposition.",
        "A 29-year-old woman has galactorrhea, secondary amenorrhea, visual bitemporal superior quadrantanopsia, and serum prolactin 260 ng/mL. Contrast hook effect artifact, stalk effect disconnection, cabergoline D2 agonist tumor shrinkage, and transsphenoidal resection triggers.",
        "Explain classic vs non-classic 21-hydroxylase congenital adrenal hyperplasia steroidogenic bottlenecks, elevated 17-hydroxyprogesterone screening, salt-wasting mineralocorticoid shock in newborns, and glucocorticoid suppression.",
        "Contrast hypothyroidism myxedema coma hypothermia/hyponatremia emergency management (IV levothyroxine + stress-dose hydrocortisone before T4!) against euthyroid sick non-thyroidal illness lab triad.",
        "A 46-year-old man presents with nephrolithiasis, refractory duodenal ulcers sparing typical H. pylori patterns, fasting gastrin 950 pg/mL, and gastric pH 1.5. Walk through secretin stimulation test paradox and MEN1 parathyroid/pituitary/pancreatic screening.",
        "Compare polycystic ovary syndrome Rotterdam criteria, LH:FSH hyperandrogenic theca stimulation, hepatic SHBG suppression from hyperinsulinemia, letrozole ovulation induction vs clomiphene, and endometrial cancer unopposed estrogen prophylaxis.",
        "Explain bone mineral metabolism in hypoparathyroidism (post-thyroidectomy, DiGeorge 22q11.2, pseudohypoparathyroidism GNAS imprinting Albright osteodystrophy) and Chvostek/Trousseau neuromuscular excitability.",
        "A 38-year-old woman presents six months postpartum with weight loss followed by profound fatigue, bradycardia, TSH 42 mIU/L, and undetectable radioiodine uptake. Compare painless postpartum thyroiditis biphasic course versus factitious thyrotoxicosis (low thyroglobulin).",
        "Detail perioperative insulin management for a patient with type 1 diabetes undergoing major abdominal surgery, distinguishing basal-bolus ketosis prevention from sliding-scale pitfalls.",
        "Contrast Turner syndrome (45,X coarctation, horseshoe kidney, streak gonads), Klinefelter syndrome (47,XXY seminiferous hyalinization), androgen insensitivity (46,XY elevated testosterone/LH), and 5-alpha-reductase deficiency.",
        "Explain diabetes insipidus versus psychogenic polydipsia versus osmotic diuresis workup, Copeptin assay utility replacing indirect AVP bioassays, and hyperosmolar correction curves.",
        "A 64-year-old woman has fasting neuroglycopenic Whipple triad spells with serum glucose 38 mg/dL, insulin 24 uIU/mL, C-peptide 4.8 ng/mL, and negative sulfonylurea screen. Trace localization via endoscopic ultrasound and selective arterial calcium stimulation.",
        "Compare carcinoid syndrome serotonin/kallistatin flush and right-sided tricuspid regurgitation heart disease, VIPoma Verner-Morrison secretory diarrhea, and glucagonoma necrolytic migratory erythema.",
        "Explain hormone replacement therapy timing hypothesis in premature ovarian insufficiency versus postmenopausal vasomotor symptoms, breast vs VTE risk profiles of transdermal versus oral estradiol.",
        # Infectious Disease (14)
        "Compare infective osteomyelitis contiguous spread vs hematogenous vertebral S. aureus MRI bone marrow edema, 6-week targeted antimicrobial biofilm penetration (rifampin hardware role), and surgical debridement.",
        "A 32-year-old veterinarian develops undulant fevers, drenching sweats, sacroiliitis, and small gram-negative coccobacilli growing after 5 days in blood cultures. Compare Brucella, Francisella tularensis, and Coxiella Q-fever endocarditis.",
        "Explain antimicrobial stewardship concepts: inoculum effect, post-antibiotic effect, extended-infusion beta-lactam time-above-MIC optimization, and carbapenem-sparing piperacillin-tazobactam caveats (MERINO trial).",
        "A 59-year-old splenectomized man presents 8 hours after a minor dog bite with septic shock, purpura fulminans, and spindle-shaped gram-negative rods inside neutrophils. Detail Capnocytophaga canimorsus, Pasteurella multocida, and rabies post-exposure prophylaxis categories.",
        "Compare histoplasmosis, blastomycosis (broad-based budding), coccidioidomycosis (endospore-filled spherules), and Cryptococcus neoformans India ink/CrAg meningitis induction with liposomal amphotericin B plus flucytosine.",
        "Explain Chagas disease Trypanosoma cruzi megaesophagus/cardiomyopathy pathogenesis, neurocysticercosis albendazole plus corticosteroid edema control, and Strongyloides stercoralis hyperinfection risk prior to steroids.",
        "A 27-year-old sexually active man develops painful dysuria, swollen knee monoarthritis, and tenosynovitis with 4 pustular acral skin lesions. Explain disseminated gonococcal infection complement deficiency link vs Chlamydia reactive arthritis triad.",
        "Contrast invasive group A streptococcal necrotizing fasciitis (pain out of proportion, LRINEC pitfalls, emergent surgical exploration, clindamycin ribosomal exotoxin shutdown) against clostridial myonecrosis.",
        "Explain pre-exposure prophylaxis (tenofovir-emtricitabine vs long-acting cabotegravir), post-exposure needle-stick 3-drug timing within 72 hours, and drug-resistance M184V vs K65R reverse transcriptase mutations.",
        "Compare primary, secondary, early/late latent, and neurosyphilis diagnostic algorithms (treponemal CIA/FTA-ABS vs non-treponemal RPR titers), Jarisch-Herxheimer reaction physiology, and penicillin desensitization in pregnancy.",
        "A 41-year-old kidney donor recipient develops fever, leukopenia, atypical lymphocytosis, transaminitis, and retinitis 3 months post-transplant after stopping valganciclovir prophylaxis. Outline CMV antigenemia/PCR viral load monitoring and UL97 ganciclovir resistance mutation testing.",
        "Explain Babesia microti tick-borne intraerythrocytic Maltese-cross tetrad hemolysis and atovaquone-azithromycin versus quinine-clindamycin exchange transfusion criteria.",
        "Compare mucormycosis ketoacidotic rhino-orbital-cerebral broad ribbon-like non-septate 90-degree hyphae angioinvasion against invasive Candida endocarditis echinocandin susceptibility.",
        "Detail tetanus neurotoxin retrograde axonal transport and VAMP-2 synaptobrevin cleavage (spastic lockjaw) versus botulinum SNAP-25 flaccid descending cranial paralysis and equine antitoxin timing.",
        # GI, Hepatology, Surgery (14)
        "Compare autoimmune hepatitis (type 1 SMA/ANA vs type 2 LKM-1, interface plasma cell hepatitis), Wilson copper overload, alpha-1 antitrypsin PAS-positive diastase-resistant globules, and hemochromatosis HFE C282Y iron saturation.",
        "A 68-year-old man develops bright red rectal bleeding, tenesmus, and left lower quadrant pain with fever 38.4 C, WBC 16k, and pericolic fat stranding with a 5 cm abscess on CT. Compare diverticulitis Hinchey classification percutaneous drainage vs ischemic colitis watershed thumbprinting.",
        "Explain esophageal motility disorders on high-resolution manometry: achalasia (integrated relaxation pressure, aperistalsis, Chagas mimic), distal esophageal spasm corkscrew swallow, and eosinophilic esophagitis ringed trachealization.",
        "Contrast small bowel obstruction adhesiolysis versus strangulated closed-loop ischemia markers, gallstone ileus Rigler triad, and Ogilvie acute colonic pseudo-obstruction neostigmine reversal (after excluding mechanical block).",
        "A 51-year-old woman with obesity presents after laparoscopic cholecystectomy with progressive jaundice, pale stools, and biloma on ultrasound. Explain Strasberg bile duct injury taxonomy, ERCP sphincterotomy/stenting vs Roux-en-Y hepaticojejunostomy.",
        "Compare hepatocellular carcinoma LI-RADS arterial phase hyperenhancement + venous washout kinetics, BCLC staging, sorafenib/atezolizumab-bevacizumab systemic therapy, and Milan transplant criteria.",
        "Explain Barrett esophagus intestinal metaplasia goblet cells, low-grade vs high-grade dysplasia endoscopic mucosal resection plus radiofrequency ablation, and achalasia squamous-cell carcinoma risk.",
        "Contrast portal hypertension pre-sinusoidal (schistosomiasis, portal vein thrombosis), sinusoidal (cirrhosis), and post-sinusoidal (Budd-Chiari hepatic vein thrombosis in polycythemia vera) ascites gradients.",
        "A 43-year-old woman on oral contraceptives has an incidental 7 cm hypervascular liver mass without central scar. Differentiate hepatic adenoma malignant rupture risk from focal nodular hyperplasia (central stellate scar) and cavernous hemangioma.",
        "Explain short bowel syndrome nutritional adaptations, bile salt malabsorption cholerheic diarrhea (colon intact) vs oxalate nephrolithiasis hyperabsorption, and teduglutide GLP-2 analog trophic effects.",
        "Compare acute mesenteric ischemia embolic superior mesenteric artery catastrophe against chronic mesenteric intestinal angina postprandial pain and food fear.",
        "Explain pancreatic ductal adenocarcinoma CA 19-9 caveats in Lewis antigen-negative patients, double duct sign on CT/MRCP, Whipple pancreaticoduodenectomy margin anatomy, and FOLFIRINOX neoadjuvant benefit.",
        "Contrast peptic stricture, Schatzki mucosal ring dysphagia, Zenker esophageal hypopharyngeal Killian triangle pulsion diverticulum, and Plummer-Vinson esophageal web.",
        "A 30-year-old trauma patient has pelvic fracture and persistent shock despite 2 L crystalloid and binder placement, with positive peritoneal FAST fluid. Outline ATLS damage-control resuscitation 1:1:1 RBC:FFP:platelets and preperitoneal packing.",
        # Heme, Onc, Rheum, Derm, Ophtho, Toxicology (40 -> total 140)
        "Compare polycythemia vera (JAK2 V617F suppressed EPO), essential thrombocythemia (CALR/MPL), and primary myelofibrosis teardrop dacrocytes plus dry bone marrow aspirate.",
        "A 64-year-old man develops painless cervical lymphadenopathy, night sweats, weight loss >10%, and binucleated CD15+/CD30+ Reed-Sternberg cells. Detail Ann Arbor staging, ABVD bleomycin pulmonary toxicity, and brentuximab vedotin ADC target.",
        "Contrast Burkitt lymphoma c-MYC t(8;14) starry-sky kinetics, diffuse large B-cell lymphoma BCL6/BCL2 double-hit, follicular t(14;18) indolent waxing, and mantle cell cyclin D1 t(11;14).",
        "Explain paroxysmal nocturnal hemoglobinuria PIGA glycosylphosphatidylinositol anchor loss (CD55/CD59 flow cytometry, morning hemoglobinuria, hepatic vein thrombosis, ravulizumab C5 blockade).",
        "Compare hereditary spherocytosis ankyrin/spectrin osmotic fragility EMA-binding defect, hereditary elliptocytosis, and pyruvate kinase deficiency ATP depletion hemolysis.",
        "A 48-year-old woman with severe menorrhagia has normal PT, prolonged aPTT normalizing on mixing, bleeding time 14 min, and factor VIII 32%. Distinguish Type 1/2A/2B/3 von Willebrand disease and desmopressin (DDAVP) contraindications.",
        "Explain CAR-T cell axicabtagene ciloleucel cytokine release syndrome (tocilizumab IL-6R blockade) versus immune effector cell-associated neurotoxicity syndrome (ICANS dexamethasone priority).",
        "Contrast acute lymphoblastic leukemia CALLA CD10+ CNS/testicular sanctuary relapse against acute myeloid leukemia FLT3-ITD/IDH1/NPM1 targetable cytogenetics.",
        "A 70-year-old man has painless macrocytic anemia MCV 118 fL, hypersegmented neutrophils, elevated methylmalonic acid and homocysteine, and subacute combined dorsal/lateral column degeneration. Explain pernicious anemia intrinsic factor vs nitrous oxide methionine synthase inactivation.",
        "Explain iron overload pathophysiology comparing hereditary hemochromatosis hepatic phlebotomy against transfusion hemosiderosis deferasirox oral iron chelation.",
        "Compare systemic sclerosis diffuse anti-Scl-70 pulmonary fibrosis versus limited CREST anti-centromere calcinosis/PAH, Sjogren anti-Ro/La dry eye focus score, and mixed connective tissue disease anti-U1 RNP.",
        "A 38-year-old woman presents with proximal hip and shoulder weakness, Gottron violaceous papules over MCP joints, heliotrope eyelid rash, and CK 4,800 U/L. Distinguish dermatomyositis perimysial CD4/complement pathology and occult malignancy screening from inclusion body myositis.",
        "Contrast ankylosing spondylitis HLA-B27 bamboo spine syndesmophytes, uveitis, and IL-17/TNF inhibition against fibromyalgia central sensitization non-inflammatory tender points.",
        "Compare IgA vasculitis (Henoch-Schonlein palpable purpura, arthralgia, mesangial IgA deposition), polyarteritis nodosa (HBV medium-vessel renal/mesenteric microaneurysms sparing lungs), and Takayasu pulseless arch disease.",
        "A 55-year-old man has symmetric polyarthritis, subcutaneous nodules, splenomegaly, and ANC 420/uL with Felty syndrome. Contrast methotrexate folate antagonism + hepatotoxicity prophylaxis against leflunomide DHODH teratogenic washout.",
        "Compare antiphospholipid syndrome lupus anticoagulant paradox (prolonged aPTT failing 1:1 mix!), anticardiolipin, anti-beta2-glycoprotein I persistent 12-week confirmation, and why warfarin outperforms DOACs in triple-positive arterial APS.",
        "Explain Behcet disease recurrent oral/genital aphthous ulcers, pathergy hypersensitivity, hypopyon panuveitis, and HLA-B51 venous thromboses.",
        "Contrast pemphigus vulgaris (intraepidermal desmoglein-3 IgG acantholysis, positive Nikolsky sign, flaccid bullae sparing palms) versus bullous pemphigoid (subepidermal hemidesmosome BP180 linear BMZ IF, tense blisters).",
        "A 29-year-old patient develops dusky tender plaques, mucosal hemorrhagic crusting across 25% body surface area, and epidermal detachment 14 days after starting allopurinol. Contrast DRESS eosinophilic organ overlap against Stevens-Johnson syndrome / TEN granulysin keratinocyte apoptosis.",
        "Compare basal cell carcinoma pearly rolled telangiectatic borders vs squamous cell keratin pearls vs melanoma Breslow depth sentinel lymph node biopsy and BRAF V600E/MEK targeted combinations.",
        "Explain erythema nodosum septal panniculitis triggers (sarcoidosis, IBD, streptococcus, OCPs) versus leukocytoclastic vasculitis dermal postcapillary venule fibrinoid necrosis and pyoderma gangrenosum pathergy.",
        "Compare central retinal artery occlusion (cherry-red spot, carotid amaurosis fugax, ocular massage window) versus central retinal vein occlusion (blood-and-thunder dilated tortuous veins) versus retinal detachment curtain.",
        "A 65-year-old woman reports progressive bilateral central visual distortion where straight doorframe lines appear wavy on Amsler grid testing. Differentiate dry geographic-atrophy drusen from wet choroidal neovascularization anti-VEGF ranibizumab/aflibercept.",
        "Contrast episcleritis (phenylephrine blanching, painless injection) against necrotizing scleritis (boring nocturnal pain, systemic RA/GPA association), anterior uveitis perilimbal flush, and corneal herpes simplex dendritic ulcer (topical steroid danger!).",
        "A 19-year-old college student ingests windshield washer fluid, presenting 14 hours later with visual snow, anion gap 26, serum osmolality 338 mOsm/kg (calculated 295). Explain methanol formate optic papillitis, alcohol dehydrogenase fomepizole blockade, and folic acid cofactor therapy.",
        "Detail tricyclic antidepressant overdose QRS >100 ms / R wave in aVR widening pharmacology, why sodium bicarbonate alkalinization narrows QRS and prevents torsades/seizures, and physostigmine contraindication.",
        "Compare carbon monoxide left-shifted carboxyhemoglobin hypoxemia (delayed basal ganglia leukoencephalopathy), methemoglobinemia chocolate blood/saturation gap 85% + methylene blue NADPH reductase, and cyanide hydroxocobalamin complexation.",
        "Explain beta-blocker versus calcium-channel blocker severe cardiogenic overdose physiology: glucagon Gs cAMP activation bypassing beta receptors versus high-dose insulin euglycemia hyperinsulinemic metabolic rescue.",
        "Compare acute organophosphate muscarinic-nicotinic cholinergic crisis (bronchorrhea, miosis, bradycardia 38 bpm, fasciculations) against acute digoxin toxicity (yellow-green halos, hyperkalemia lethality correlation, and Digibind Fab fragment indications).",
        "Compare ethylene glycol calcium oxalate monohydrate envelope crystals and hypocalcemic tetany against isopropyl alcohol osmolar gap ketosis WITHOUT metabolic acidosis.",
        "Detail gestational diabetes 1-hour 50g screen vs 3-hour 100g OGTT insulin resistance mediated by human placental lactogen, macrosomia brachial plexus dystocia, and neonatal transient hyperinsulinemic hypoglycemia.",
        "Compare placenta previa painless bright bleeding against placental abruption painful tetanic uterus + DIC risk, vasa previa fetal vessel rupture, and uterine atony postpartum uterotonics (oxytocin, carboprost asthma precaution, methylergonovine hypertension).",
        "A 14-day-old newborn presents with umbilical cord detachment delay, leukocytosis WBC 64,000/uL without pus, and gingivitis. Contrast leukocyte adhesion deficiency CD18 integrin defect, chronic granulomatous disease NADPH oxidase nitroblue tetrazolium/DHR failure, and Chediak-Higashi lysosomal granules.",
        "Explain Kawasaki disease mucocutaneous coronary aneurysm vasculitis (5-day fever + 4/5 CRASH criteria, incomplete infant risk, IVIG + high-dose aspirin Reye monitoring) vs MIS-C.",
        "Compare croup (parainfluenza steeple sign, inspiratory stridor, dexamethasone + racemic epinephrine) versus bacterial epiglottitis (thumbprint sign, drooling tripod) versus bronchiolitis RSV hydration/suctioning.",
        "A 6-week-old boy has worsening jaundice since week 2, acholic pale stools, dark urine, direct bilirubin 5.8 mg/dL, and absent gallbladder on fasting ultrasound. Detail biliary atresia Kasai hepatoportoenterostomy 60-day critical window.",
        "Explain congenital adrenal hyperplasia versus congenital hypothyroidism newborn screening, galactosemia E. coli sepsis and cataract accumulation, and phenylketonuria tetrahydrobiopterin deficiency cofactor distinction.",
        "Compare teratoma alpha-fetoprotein/beta-hCG testicular germ cell tumors (seminoma radiosensitive PLAP+ vs yolk sac Schiller-Duval vs choriocarcinoma trophoblastic hemorrhage) and BEP bleomycin/cisplatin toxicity.",
        "Explain oncologic emergencies: superior sulcus Pancoast tumor sympathetic chain compression, spinal cord epidural compression immediate dexamethasone before MRI, and symptomatic hypercalcemia zoledronate/calcitonin bridge.",
        "Detail delirium versus dementia diagnostic CAM criteria in hospitalized older adults, anticholinergic Beers criteria avoidance, and nonpharmacologic multicomponent orientation protocols over routine haloperidol.",
    ]

    chem = [
        # Physical, Quantum, Statistical Thermo, Spectroscopy (24)
        "Derive the particle-in-a-1D-box normalized wavefunctions psi_n(x) and energy eigenvalues E_n, compute the expectation values <x> and <x^2> for n=2, and relate selection rules for dipole transitions to box length conjugation in cyanine dyes.",
        "Explain the harmonic oscillator zero-point energy, Hermite polynomial nodal structure, selection rule Δv = ±1, and how the Morse potential anharmonicity constant chi_e modifies hot bands and overtone spacings to reveal dissociation energy D_0.",
        "Compare rigid rotor rotational spectroscopy (B = h/(8π²cI), centrifugal distortion D_J, population Boltzman max J_max) with symmetric-top vs asymmetric-top selection rules and microwave isotopic bond-length determination in CO.",
        "Derive the Boltzmann distribution from maximizing multiplicity ln W subject to constant particle number N and total energy U, define the canonical partition function Q(N,V,T), and obtain Sackur-Tetrode ideal gas translational entropy.",
        "Explain vibrational and rotational partition functions of heteronuclear versus homonuclear diatomics (symmetry number sigma=2, ortho/para nuclear spin statistics of H2) and why equipartition fails for vibrational heat capacity at 300 K.",
        "Derive the Clapeyron and Clausius-Clapeyron equations from chemical potential equality μ_α(P,T) = μ_β(P,T) across a phase boundary, and calculate the vapor pressure of ethanol at 50 C given ΔH_vap = 38.6 kJ/mol.",
        "Compare Helmholtz free energy A and Gibbs free energy G Legendre transforms, prove the Maxwell relation (∂S/∂V)_T = (∂P/∂T)_V, and derive the thermodynamic equation of state (∂U/∂V)_T = T(∂P/∂T)_V - P.",
        "Explain partial molar volumes, Gibbs-Duhem relation x_1 dμ_1 + x_2 dμ_2 = 0, regular solution mixing enthalpy ΔH_mix = χ x_1 x_2, and lower vs upper critical solution temperatures in binary liquid mixtures.",
        "Derive the steady-state Michaelis-Menten and Lindemann-Hinshelwood unimolecular pressure-falloff rate laws, demonstrating first-order limit at high [M] and second-order collisional activation limit at low [M].",
        "Explain Marcus electron-transfer theory: inner-shell vs outer-shell solvent reorganization energy lambda, parabolic diabatic free energy curves, and why the inverted region occurs when -ΔG° > lambda.",
        "Compare steady-state fluorescence Stern-Volmer collisional quenching vs static ground-state complex quenching plots, Forster resonance energy transfer (FRET) r^-6 distance scaling, and Jablonski spin-orbit intersystem crossing.",
        "Explain the Franck-Condon principle using vertical electronic transitions between displaced ground and excited potential energy surfaces, vibrational progressions, and mirror-image absorption-emission symmetry.",
        "Compare Woodward-Hoffmann frontier molecular orbital (HOMO/LUMO) correlation diagrams for electrocyclic [4n] conrotatory vs [4n+2] disrotatory thermal ring closures and [2+2] photochemical dimerization.",
        "Explain Huckel molecular orbital pi-energy determinants for linear conjugated polyenes (butadiene vs hexatriene) versus cyclic Frost circle stability diagrams (4n+2 aromatic vs 4n antiaromatic).",
        "Compare UV-Vis electronic transitions (n -> pi* vs pi -> pi* solvent red/blue shifts), infrared group frequencies of anhydrides/lactones/amides, and Fermi resonance doublet splitting.",
        "Explain 2D NMR COSY, HSQC, and HMBC cross-peak correlations, NOESY through-space 1/r^6 dipolar magnetization transfer, and dynamic variable-temperature coalesce rate determination k_c = pi Δν / sqrt(2).",
        "Compare electron ionization (EI) radical-cation fragmentation (alpha-cleavage, retro-Diels-Alder), electrospray soft ionization (ESI) multiply charged protein envelopes, and MALDI-TOF mass resolution.",
        "Explain X-ray single-crystal diffraction Bragg's law, reciprocal lattice Ewald sphere construction, Fourier electron density maps, and the crystallographic phase problem (anomalous dispersion SAD/MAD).",
        "Describe BET (Brunauer-Emmett-Teller) multilayer adsorption isotherms vs Langmuir monolayer kinetics, pore-size BJH hysteresis loops in mesoporous silica, and surface area determination via N2 at 77 K.",
        "Explain ultrafast pump-probe transient absorption spectroscopy, femtosecond stimulated emission vs excited-state absorption, and conical intersection nonadiabatic internal conversion.",
        "Compare Debye-Huckel limiting law activity coefficients ln γ_± = -A |z+z-| sqrt(I), extended Davies corrections, and Onsager reciprocal conductance deviations with ionic strength.",
        "Explain electrochemical cyclic voltammetry: Randles-Sevcik peak current i_p scaling with ν^(1/2), reversible 59/n mV peak-to-peak separation vs quasi-reversible Butler-Volmer transfer coefficient alpha.",
        "Derive the osmotic virial expansion, Flory-Huggins polymer solution mixing entropy reduction, and light-scattering Zimm plot determination of weight-average molar mass M_w and radius of gyration R_g.",
        "Explain laser-induced breakdown and atomic absorption spectroscopy versus ICP-OES/ICP-MS trace elemental quantification, Boltzmann excited-state population thermal sensitivity, and matrix internal standardization.",
        # Inorganic, Coordination, Solid-State, Electrochem & Analytical (24)
        "Apply Wade-Mingos polyhedral skeletal electron pair theory (closo n+1, nido n+2, arachno n+3) to predict structures of B6H6^2-, B5H9, and transition-metal carbonyl cluster [Co6(CO)16].",
        "Apply the 18-electron rule and isolobal analogy to classify Fe(CO)5, [(eta5-C5H5)Fe(CO)2]2, Cr(eta6-C6H6)2, and Vaska's complex trans-[IrCl(CO)(PPh3)2] oxidative addition stereochemistry.",
        "Compare pi-acceptor synergic backbonding in metal carbonyls (CO stretching frequency shift across [V(CO)6]-, Cr(CO)6, [Mn(CO)6]+), Tolman cone angle steric hindrance, and electronic Tolman parameter on phosphines.",
        "Explain Wilkinson's catalyst [RhCl(PPh3)3] homogeneous alkene hydrogenation mechanism, Noyori asymmetric transfer hydrogenation, and Grubbs/Schrock olefin metathesis metallacyclobutane intermediates.",
        "Compare Monsanto rhodium vs Cativa iridium methanol carbonylation cycles, showing oxidative addition of CH3I, migratory CO insertion, and reductive elimination of acetyl iodide.",
        "Explain Tanabe-Sugano d3 and d6 octahedral correlation diagrams, Racah interelectronic repulsion B parameter nephélauxetic reduction, spin-allowed Laporte-forbidden d-d extinction coefficients vs MLCT bands.",
        "Contrast outer-sphere Marcus electron transfer with Taube inner-sphere halide-bridged redox mechanism using [Co(NH3)5Cl]2+ and [Cr(H2O)6]2+ substitution-lability vs inertness proof.",
        "Explain lanthanide contraction f-orbital poor shielding, microstate spin-orbit coupled Russell-Saunders J ground terms (^7F_0 for Eu3+, ^8S_7/2 for Gd3+ MRI relaxivity), and narrow f-f luminescence lines.",
        "Compare ionic crystal structures (rocksalt NaCl 6:6, CsCl 8:8, zinc blende 4:4, fluorite CaF2 8:4, perovskite ABO3 Goldschmidt tolerance factor), radius-ratio rules, and Schottky vs Frenkel defects.",
        "Explain solid-state band theory in metals, intrinsic Si/Ge, p-type vs n-type doped semiconductors, optical band-gap Tauc plots, and copper-oxide vs iron-pnictide superconductor Meissner characteristics.",
        "Compare fuel cell proton exchange membrane (PEMFC) hydrogen electrocatalysis, solid oxide YSZ O2- ion-conducting electrolytes at 800 C, and supercapacitor electrical double-layer vs pseudocapacitance storage.",
        "Derive liquid junction potentials, ion-selective glass pH membrane alkaline error and acid error, and potentiometric titration Gran plot equivalence point linearization.",
        "Compare EDTA complexometric back-titration with masking agents (CN-, F-, triethanolamine), metallochromic indicator Eriochrome Black T pH 10 Mg/Ca selectivity, and conditional formation constants K'_f.",
        "Explain Karl Fischer coulometric and volumetric trace moisture titration stoichiometry involving SO2, I2, imidazole base, and methanol alkyl sulfite formation.",
        "Contrast inductively coupled plasma mass spectrometry collision/reaction cell polyatomic interference suppression, triple-quadrupole MRM selectivity, and isotope-dilution calibration precision.",
        "Explain thermogravimetric analysis (TGA), differential scanning calorimetry (DSC) glass transition T_g step vs melting endotherm peak, and crystallization exotherm kinetics.",
        "Compare capillary electrophoresis electroosmotic flow profile (plug flow vs parabolic HPLC laminar flow), micellar electrokinetic chromatography for neutral analytes, and Taylor dispersion band broadening.",
        "Explain supercritical fluid chromatography CO2-cosolvent miscibility tuning, size-exclusion chromatography hydrodynamic volume calibration against polystyrene standards, and HILIC hydrophilic partitioning.",
        "Compare Pourbaix E_h-pH diagrams for iron corrosion (immunity, passivation Fe2O3/Fe3O4, active Fe2+ dissolution), sacrificial magnesium anodes, and overpotential Tafel kinetics.",
        "Describe bioinorganic metalloenzyme active sites: nitrogenase FeMo-cofactor P-cluster ATP reduction, carbonic anhydrase Zn2+ hydroxide nucleophile, and cytochrome P450 Compound I oxoferryl porphyrin radical.",
        "Explain zeolite aluminosilicate Si/Al Bronsted acid framework catalysis, shape-selective ZSM-5 xylene isomerization, and metal-organic framework (MOF) breathing isotherm adsorption.",
        "Compare photochemical water splitting semiconductor band-edge alignment vs NHE water oxidation/reduction potentials, TiO2 electron-hole recombination, and cocatalyst overpotential lowering.",
        "Explain magnetic susceptibility Gouy/Evans balance determination, Curie-Weiss temperature corrections, spin-crossover Fe(II) thermal bistability, and SQUID single-molecule magnet blocking.",
        "Compare sol-gel alkoxide hydrolysis-condensation silica aerogel synthesis, hydrothermal nanocrystal Ostwald ripening, and atomic layer deposition self-limiting surface half-reactions.",
        # Organic Mechanisms, Synthesis & Stereochemistry (32 -> total 80)
        "Contrast Zimmerman-Traxler chair transition states in E(O) vs Z(O) boron enolate aldol additions, showing how 1,3-diaxial allyl/methyl clashes control syn versus anti diastereoselectivity.",
        "Explain Evans oxazolidinone chiral auxiliary alkylations and syn-aldol acylations: chelation control vs dipole-minimized non-chelated conformations and oxidative LiOOH auxiliary cleavage.",
        "Compare kinetic enolate generation (LDA, -78 C, THF, aprotic) versus thermodynamic enolization (NaOEt/EtOH or KH/rt), Stork enamine monoalkylation, and silyl enol ether Mukaiyama aldol reactions.",
        "Describe Shapiro and Bamford-Stevens tosylhydrazone decompositions, Corey-Seebach 1,3-dithiane umpolung acyl anion equivalents, and Nef nitroalkane-to-carbonyl hydrolysis.",
        "Compare Sharpless asymmetric epoxidation (Ti(OiPr)4, (+)/(-)-DET, TBHP on allylic alcohols), Jacobsen-Katsuki salen epoxidation of cis-alkenes, and Shi fructose-ketone dioxirane catalysis.",
        "Explain Robinson annulation sequence (Michael addition followed by intramolecular aldol condensation), Hajos-Parrish-Eder-Sauer-Wiechert proline organocatalysis, and Wieland-Miescher ketone synthesis.",
        "Compare [3,3]-sigmatropic Claisen, Ireland-Claisen (silyl ketene acetal E/Z control), Cope, and oxy-Cope rearrangements alongside [2,3]-Wittig stereospecificity.",
        "Trace all intermediates in Swern (DMSO, (COCl)2, Et3N, -78 C), Dess-Martin periodinane, Ley-Griffith TPAP/NMO, and TEMPO/bleach alcohol oxidations without overoxidation.",
        "Compare selective palladium/nickel/copper couplings: Stille (organotin toxicity vs mildness), Negishi (organozinc), Buchwald-Hartwig C-N amination, and Ullmann/Chan-Lam C-O/C-N arylations.",
        "Explain Criegee mechanism of alkene ozonolysis (molozonide -> carbonyl oxide -> primary ozonide) and how reductive (Me2S, PPh3, NaBH4) vs oxidative (H2O2) workups diverge.",
        "Contrast Birch dissolving-metal reduction (Na/NH3(l)/t-BuOH) regioselectivity on anisole (electron-donating -> 1,4-diene avoiding ipso carbanion) vs benzoic acid (electron-withdrawing -> conjugated cyclohexadiene).",
        "Compare Corey-Bakshi-Shibata (CBS) oxazaborolidine ketone reduction, Midland Alpine-Borane propargyl reduction, and Luche CeCl3/NaBH4 selective 1,2-reduction of enones.",
        "Explain radical chain cyclizations obeying Baldwin's rules (5-exo-trig dominance via SOMO-LUMO Burgi-Dunitz trajectory), Barton-McCombie xanthate deoxygenation with Bu3SnH vs TTMSS, and Giese radical addition.",
        "Compare carbene singlet vs triplet spin states, Skell-Woodworth stereospecific cis/trans alkene cyclopropanation, Simmons-Smith Zn-Cu carbenoid hydroxymethyl direction, and Wolff rearrangement of alpha-diazoketones.",
        "Explain Mitsunobu inversion (DEAD/DIAD, PPh3, pKa < 11 pronucleophile oxyphosphonium betaine) and why poorly acidic alcohols fail without betaine deprotonation.",
        "Compare Peterson olefination acidic syn- vs basic anti-beta-hydroxysilane elimination, Julia-Kocienski benzothiazolyl sulfone E-olefination, and Tebbe/Petasis methylenation of esters.",
        "Trace heteroaromatic electrophilic substitution regioselectivity at C2 vs C3 across furan, thiophene, pyrrole, and indole, contrasted with Chichibabin C2 nucleophilic amination of pyridine.",
        "Compare Fischer indole synthesis ([3,3]-sigmatropic hydrazone rearrangement), Skraup/Friedlander quinoline syntheses, Paal-Knorr pyrrole/furan synthesis, and Hantzsch dihydropyridine condensation.",
        "Explain Barton nitrite photolysis remote C-H delta-functionalization, Hofmann-Loffler-Freytag pyrrolidine synthesis, and Norrish Type I alpha-cleavage vs Type II gamma-hydrogen abstraction.",
        "Contrast stereochemical retention vs inversion vs neighboring-group anchimeric assistance (beta-bromonium, trans-acetoxy cyclohexyl tosylate, phenonium bridged ion solvolysis rates).",
        "Explain Felkin-Anh vs Cram-chelate asymmetric carbonyl addition models, Burgi-Dunitz 107-degree nucleophilic trajectory, and 1,3-asymmetric induction Evans-Tishchenko rules.",
        "Compare ring-closing metathesis (RCM), cross-metathesis reactivity types, and ring-opening metathesis polymerization (ROMP) strained norbornene thermodynamic driving force.",
        "Describe photoredox catalysis using [Ru(bpy)3]2+ and Ir(ppy)3 single-electron oxidation/reduction potentials, decarboxylative Giese additions, and metallaphotoredox C(sp2)-C(sp3) cross-coupling.",
        "Explain nucleophilic fluorination (DAST deoxyfluorination, Deoxo-Fluor) versus electrophilic Selectfluor alpha-fluorination and how CF3 substitution modulates pKa, lipophilicity, and CYP450 metabolic stability.",
        "Compare macromolecular synthesis: reversible addition-fragmentation chain-transfer (RAFT), atom-transfer radical polymerization (ATRP), and ring-opening polymerization of lactide/caprolactone.",
        "Explain atropisomerism axial chirality in BINOL/BINAP ortho-substituted biaryls, planar chirality in ferrocenes/trans-cyclooctene, helical hexahelicene optical activity, and racemization barrier kinetics.",
        "Compare Baylis-Hillman DABCO catalysis, Henry nitro-aldol, Mannich iminium condensation, Strecker amino acid synthesis, and Ugi four-component isocyanide multicomponent cascade.",
        "Describe protecting group orthogonal deprotection matrix for alcohols (TBDMS/TIPS/THP/Bn/PMB) and amines (Boc TFA / Fmoc piperidine / Cbz H2-Pd / Teoc fluoride).",
        "Explain Curtius acyl azide, Lossen hydroxamate, and Schmidt hydrazoic acid rearrangements, emphasizing concerted retention of configuration at the migrating stereocenter.",
        "Compare hydroformylation rhodium/cobalt oxo catalysis linear-to-branched aldehyde regioselectivity tuned by bite-angle diphosphines (Xantphos) and Wacker PdCl2/CuCl2 aerobic ethylene oxidation.",
        "Explain kinetic isotope effects (k_H/k_D ≈ 6-7 primary zero-point energy difference vs alpha/beta secondary sp2-sp3 rehybridization) to map concerted E2 vs stepwise E1cb mechanisms.",
        "Compare pericyclic [1,5]-hydride shifts (antarafacial vs suprafacial), ene reactions, 1,3-dipolar Huisgen azide-alkyne cycloaddition (thermal vs CuAAC regioselective 1,4-triazole), and strain-promoted SPAAC.",
    ]

    bio = [
        # Structural, Biophysics, Enzymology, Proteins (20)
        "Explain Ramachandran phi/psi sterically allowed torsion angle basins for right-handed alpha-helices, parallel vs antiparallel beta-sheets, left-handed collagen polyproline II helices, and glycine/proline exceptions.",
        "Describe cryo-electron microscopy (cryo-EM) vitrification, direct electron detector motion correction, 2D class averaging, Fourier shell correlation (FSC=0.143) resolution estimation, and atomic model B-factor refinement.",
        "Compare allosteric cooperative models: Monod-Wyman-Changeux (MWC concerted R/T states without hybrids) versus Koshland-Nemethy-Filmer (KNF sequential induced fit permitting negative cooperativity) with Scatchard plots.",
        "Explain pre-steady-state stopped-flow enzyme burst kinetics for chymotrypsin acyl-enzyme formation vs release, catalytic efficiency k_cat/K_m diffusion limits (~10^8-10^9 M^-1 s^-1), and triosephosphate isomerase perfection.",
        "Compare intrinsically disordered proteins (IDPs) phase separation via multivalent weak sticker-spacer interactions, biomolecular condensates (stress granules, nucleolus), and prion-like low-complexity domains.",
        "Describe Sec61 cotranslational SRP translocation across the rough ER membrane, tail-anchored TRC40/GET3 targeting, and N-linked dolichol-oligosaccharide calnexin/calreticulin quality control.",
        "Explain nuclear pore complex FG-nucleoporin selective barrier, Ran-GTP/Ran-GDP spatial gradient maintained by chromatin-bound RCC1 and cytoplasmic RanGAP, and importin/exportin directional cargo release.",
        "Compare kinesin plus-end processive hand-over-hand ATP stepping, cytoplasmic dynein AAA+ ring minus-end motility with dynactin, and myosin II non-processive sarcomere power strokes.",
        "Explain actin treadmilling ATP-cap polarity, Arp2/3 branched nucleation (70-degree angles at lamellipodia) vs formin processive barbed-end elongation, and microtubule dynamic instability GTP-cap catastrophe.",
        "Describe viral entry fusion machinery: Class I HIV gp41/influenza HA six-helix bundle conformational snap vs Class II flavivirus pH-triggered endosomal dimer-to-trimer insertion.",
        "Compare mechanosensitive Piezo1/2 ion channel curved dome gating, voltage-gated K+ S4 arginine paddle coupling with selectivity filter TVGYG carbonyl dehydration, and TRP nociceptive polymodal activation.",
        "Explain ABC transporter alternating-access ATP-binding cassette dimerization (CFTR chloride channel phosphorylation gating vs P-glycoprotein multidrug efflux) vs proton-antiporter Major Facilitator Superfamily.",
        "Describe amyloid fibril cross-beta steric zipper polymorphism across Abeta42, tau paired helical filaments, and alpha-synuclein Lewy neurites, plus cryo-EM seeding nucleation kinetics.",
        "Compare isothermal titration calorimetry (ITC) single-experiment ΔH, ΔS, K_d, and stoichiometry determination with surface plasmon resonance (SPR) real-time k_on/k_off kinetics and microscale thermophoresis.",
        "Explain bacteriophage lambda lytic versus lysogenic bistable genetic switch: CI repressor cooperative binding at O_R1/O_R2/O_R3 versus Cro competition and RecA-mediated SOS induction.",
        "Describe two-component bacterial histidine kinase autophosphorylation and aspartate response-regulator phosphorelay alongside LuxI/LuxR quorum-sensing autoinducer threshold switches.",
        "Explain synthetic biology genetic toggle switches, repressilator 3-gene ring oscillations, coherent vs incoherent type-1 feed-forward loop pulse-generation and fold-change detection dynamics.",
        "Compare single-molecule optical tweezers DNA overstretching kinetics, smFRET conformational dynamics, and single-molecule magnetic tweezers supercoiling torque measurement on topoisomerase II.",
        "Explain directed evolution phage-assisted continuous evolution (PACE) versus yeast surface display sorting for engineering thermostable PETase hydrolases or high-affinity nanobodies.",
        "Describe cryo-focused ion beam milling and cryo-electron tomography (cryo-ET) subtomogram averaging to resolve native 80S polysomes and NPC scaffolds inside intact cells.",
        # Genetics, Genomics, Evo/Dev, Immunology, Neuro, Micro (60 -> total 80)
        "Explain RNA polymerase I (rRNA nucleolus), II (mRNA/snRNA), and III (tRNA/5S rRNA internal promoters), comparing polyadenylation cleavage CPSF/CSTF coupling and histone stem-loop non-polyA maturation.",
        "Describe topoisomerase I transient single-strand tyrosine-DNA phosphodiester swivel versus type II ATP-dependent double-strand gate passage (DNA gyrase negative supercoiling, etoposide/fluoroquinolone traps).",
        "Explain meiotic homologous recombination double-strand break formation by Spo11, resection, Dmc1 strand invasion, double Holliday junction resolution vs synthesis-dependent strand annealing (crossover vs gene conversion).",
        "Compare dosage compensation strategies: mammalian XIST long noncoding RNA PRC2 coating, Drosophila MSL male H4K16ac 2x hyperactivation, and C. elegans condensin-like hermaphrodite halving.",
        "Explain genomic structural variation detection from paired-end Illumina discordant reads vs PacBio HiFi long reads, optical genome mapping, and balanced translocation topological domain enhancer hijacking.",
        "Describe single-cell RNA-seq UMI barcoding (10x Chromium GEMs), PCA/UMAP dimensionality reduction, batch correction Harmony pitfalls, RNA velocity spliced/unspliced ratios, and spatial Visium/MERFISH trade-offs.",
        "Explain base excision by glycosylases (OGG1 8-oxoG, UNG deaminated cytosine) vs transcription-coupled NER (CSB/CSA RNA pol II stalling) vs Fanconi anemia interstrand crosslink repair.",
        "Compare transposon-mediated sleeping beauty/PiggyBac gene delivery, adeno-associated virus (AAV) inverted terminal repeat episomal persistence vs lentiviral VSV-G integration mutagenesis risks.",
        "Describe ribosome quality control pathways: no-go/nonstop mRNA dom34/Hbs1 splitting, NEMF CAT-tailing, Listerin ubiquitination, and bacterial tmRNA trans-translation rescue.",
        "Explain polycomb (PRC1 H2AK119ub / PRC2 EZH2 H3K27me3) versus Trithorax (COMPASS MLL H3K4me3) antagonistic maintenance of Hox segment identity boundaries during Drosophila embryogenesis.",
        "Compare bicoid/nanos maternal anterior-posterior morphogen gradients in Drosophila syncytial blastoderm, gap/pair-rule/segment polarity cascades, and hedgehog-wingless boundary maintenance.",
        "Explain vertebrate limb bud patterning along proximal-distal (AER FGF8), anterior-posterior (ZPA Sonic hedgehog retinoic acid gradient), and dorsal-ventral (Wnt7a/Lmx1b vs Engrailed-1) axes.",
        "Compare TGF-beta/BMP Smad2/3 vs Smad1/5/8 nuclear shuttling, canonical Wnt/beta-catenin Axin-APC-GSK3beta destruction complex inhibition, and Notch-Delta lateral inhibition gamma-secretase cleavage.",
        "Explain induced pluripotent stem cell (iPSC) Oct4/Sox2/Klf4/c-Myc epigenetic erasure vs somatic cell nuclear transfer, gastruloid organoid self-organization, and Waddington developmental landscapes.",
        "Describe floral organ identity ABCDE model (AP1, AP3/PI, AG, SEP) combinatorial MADS-box homeotic complexes predicting double-mutant sepal-only or carpel-stamen transformations in Arabidopsis.",
        "Explain molecular phylogenetics: Jukes-Cantor vs GTR substitution models, synonymous Ka/Ks purifying vs diversifying positive selection ratios, incomplete lineage sorting, and molecular clock relaxed calibration.",
        "Compare Mullerian mimicry mutual aposematism, Batesian parasitic mimicry frequency dependence, Hamilton kin selection rB > C inclusive fitness, and parent-offspring Trivers conflict.",
        "Explain speciation mechanisms: allopatric vicariance, sympatric host-shift/allopolyploidy in wheat, Dobzhansky-Muller epistatic hybrid incompatibilities, and Haldane sterility rule.",
        "Compare adaptive population genetics selective sweeps (hitchhiking diversity valley, Tajima D negative shift) versus balancing selection maintenance of MHC heterozygosity.",
        "Explain C3 vs C4 Kranz bundle-sheath PEPC spatial CO2 pump vs CAM temporal nocturnal malate storage tradeoffs under high heat, drought stomata closure, and VPD stress.",
        "Describe plant systemic acquired resistance salicylic acid NPR1-TGA immunity versus jasmonic acid/ethylene necrotrophic defense, auxin PIN polar efflux transporters, and vernalization FLC silencing.",
        "Explain nitrogenase oxygen-sensitivity protection in legume root nodules (leghemoglobin buffering, bacteroid peribacteroid membrane) and mycorrhizal phosphate-for-hexose symbiosis.",
        "Compare CRISPR-Cas adaptive prokaryotic immunity phases: adaptation (Cas1-Cas2 protospacer capture into leader array), crRNA biogenesis, and interference (Type I Cascade-Cas3 vs Type II Cas9 vs Type VI Cas13 RNA collateral cleavage).",
        "Explain bacteriophage T4 packaging headful motor, temperate lambda antitermination N/Q proteins, bacterial toxin-antitoxin persister dormancy, and conjugative F-plasmid oriT rolling-circle transfer.",
        "Describe microbial anaerobic respiration electron towers across denitrification (NO3- -> N2), sulfate reduction, methanogenesis H2/CO2 methyl-CoM reductase F430, and anammox hydrazine intermediates.",
        "Explain enveloped vs non-enveloped virus replication cycles comparing positive-sense (+)ssRNA Picornavirus subgenomic IRES translation against negative-sense (-)ssRNA Paramyxovirus RdRp stuttering polyadenylation and retroviral polypurine reverse transcription.",
        "Compare mucosal IgA dimeric J-chain transcytosis via pIgR, neonatal FcRn IgG recycling half-life extension, high-affinity IgE FcepsilonRI mast cell crosslinking, and IgG4 Fab-arm exchange tolerance.",
        "Explain NK cell missing-self activation via inhibitory KIR/CD94-NKG2A ITIMs competing against activating NKG2D MIC-A/B stress ligands and ADCC CD16 engagement.",
        "Describe thymic V(D)J recombination RSS 12/23 spacing, Artemis hairpin opening, TdT non-templated N-nucleotide addition, receptor editing, and peripheral anergy (Signal 1 without CD28-B7 Signal 2).",
        "Compare type I interferons (IFN-alpha/beta JAK1/TYK2 ISGF3 antiviral MxA/OAS/PKR) with type II IFN-gamma macrophage nitric-oxide phagolysosomal activation and Type III IFN-lambda epithelial specificity.",
        "Explain dendritic cell cDC1 BATF3 cross-presentation via phagosomal Sec61/ERAD TAP loading vs cDC2 MHC-II invariant-chain CLIP HLA-DM exchange, plus mucosal CX3CR1 sampling.",
        "Describe neuroimmune crosstalk, blood-brain barrier pericyte-astrocyte endfeet tight junctions (claudin-5), glymphatic AQP4 interstitial clearance during slow-wave sleep, and microglia TREM2 synaptic pruning.",
        "Compare NMDA ionotropic Mg2+ voltage-block coincidence detector (Ca2+ CaMKII AMPA GluA1 insertion long-term potentiation) against mGluR-mediated endocannabinoid retrograde DSI suppression.",
        "Explain presynaptic vesicle cycle: synapsin release, Munc13/18 priming, synaptotagmin Ca2+ sensor zippering VAMP2/syntaxin/SNAP25 SNARE complex, and clathrin-dynamin ultrafast endocytosis.",
        "Describe retinal phototransduction dark current closure: rhodopsin 11-cis -> all-trans isomerization, transducin PDE6 cGMP hydrolysis closing CNG channels, and OFF vs ON-bipolar mGluR6 hyperpolarization reversal.",
        "Explain cochlear hair-cell basilar membrane tonotopic traveling wave resonance, stereocilia tip-link TMC1 mechanotransduction K+ endolymphatic depolarization (+80 mV stria vascularis), and outer hair-cell prestin amplification.",
        "Compare hippocampal trisynaptic memory consolidation circuit (entorhinal -> dentate gyrus pattern separation -> CA3 recurrent autoassociation -> CA1) vs cerebellar Purkinje parallel/climbing fiber LTD motor error learning.",
        "Explain hypothalamus-pituitary neuroendocrine feedback loops, arcuate AgRP/NPY orexigenic vs POMC/CART leptin/ghrelin metabolic integration, and brown adipose UCP1 beta3-adrenergic thermogenesis.",
        "Describe renin-angiotensin-aldosterone system juxtaglomerular baroreceptor sensing, macula densa Cl- transport, ACE pulmonary bradykinin cleavage, AT1 vasoconstriction vs AT2/Mas counter-regulation.",
        "Compare oxygen-dissociation Bohr/Root effects in teleost swim bladders vs high-altitude bar-headed goose hemoglobin variants, fetal-adult gamma-to-beta globin BCL11A switching, and diving-mammal myoglobin stores.",
        "Explain erythrocyte membrane flexibility, Spectrin-Ankyrin-Band 3/4.1 cytoskeleton, Embden-Meyerhof glycolysis 2,3-BPG Rapoport-Luebering shunt, and pentose phosphate glutathione protection.",
        "Describe endothelial nitric oxide synthase shear-stress activation, guanylyl cyclase cGMP smooth muscle PKG relaxation, prostacyclin vs thromboxane platelet equilibrium, and von Willebrand shear unfolding.",
        "Explain hepatocyte zonation along portal-to-central lobular oxygen gradients (Zone 1 gluconeogenesis/urea vs Zone 3 CYP2E1/lipogenesis/glutamine synthetase), bile acid FXR enterohepatic recycling.",
        "Describe hematopoietic stem cell bone marrow endosteal vs perivascular niche CXCL12/SCF signaling, PU.1/GATA1 myeloid-erythroid cross-antagonism, and megakaryocyte polyploidization proplatelet shedding.",
        "Compare autophagy selective cargo adaptors (PINK1/Parkin mitophagy, p62 aggrephagy, LC3-II lipidation ATG5-ATG12 conjugation, mTORC1 ULK1 phosphorylation block vs AMPK activation).",
        "Explain ferroptosis iron-dependent polyunsaturated phospholipid peroxidation, GPX4 selenium-dependent detox vs System xc- cystine import inhibition (erastin) vs FSP1 CoQ10 defense.",
        "Describe cellular senescence p16INK4a/p21CIP1 cell-cycle arrest, telomere uncapping DNA damage response, NF-kappaB senescence-associated secretory phenotype (SASP), and senolytic BCL-xL/dasatinib strategies.",
        "Explain microbiome short-chain fatty acid (butyrate, propionate) HDAC inhibition in colonic Tregs, bile-acid secondary conversion, and vagal gut-brain enteroendocrine GLP-1/serotonin signaling.",
        "Describe chromatin 3D genome architecture: CTCF convergent-motif loop extrusion by cohesin, topologically associating domains (TADs), A/B compartmental phase segregation, and enhancer-promoter MED1 condensation.",
        "Explain bacteriophage-derived epigenetic defenses: bacterial restriction-modification (Type I/II/III), BREX, cyclic oligonucleotide CBASS abortive infection, and eukaryotic piRNA Piwi transposon silencing.",
        "Compare mitochondrial outer-membrane TOM/TIM23 presequence import driven by delta-psi and mtHsp70 motor vs twin-arginine translocation of folded redox proteins in thylakoids.",
        "Explain yeast mating-type switching MATa/MATalpha HO endonuclease gene conversion from silenced HML/HMR cassettes, Ste5 MAPK pheromone scaffold insulation, and Fus3 G1 arrest.",
        "Describe circadian cyanobacteria KaiABC post-translational phosphorylation oscillator reconstitutable in vitro with ATP, contrasted against mammalian cryptochrome transcriptional repression.",
        "Explain metabolic reprogramming in proliferating cancer cells (Warburg aerobic glycolysis PKM2 serine synthesis, glutaminolysis reductive carboxylation via IDH1/2, oncometabolite 2-hydroxyglutarate TET2 inhibition).",
        "Compare synthetic lethality PARP1/2 trapping in BRCA1/2 homologous-recombination-deficient tumors, ATR/CHK1 replication-stress sensitization, and WRN helicase dependency in MSI-high cancers.",
        "Explain bone homeostasis coupling between osteocyte sclerostin Wnt suppression, osteoblast Runx2/Osterix mineralization, osteoclast carbonic anhydrase II/cathepsin K resorption lacunae, and parathyroid hormone intermittent vs continuous paradox.",
        "Describe skeletal muscle fiber-type plasticity (Type I oxidative PGC-1alpha calcineurin/NFAT vs Type IIb fast glycolytic), satellite cell Pax7 asymmetric regeneration, and Duchenne dystrophin-glycoprotein sarcolemmal stability.",
        "Explain fluid-mosaic membrane lipid asymmetry: flippase/floppase ATP-dependent phosphatidylserine localization vs scramblase externalization in apoptosis/coagulation, cholesterol raft microdomains, and curvature-sensing BAR domains.",
        "Describe quorum-dependent biofilm extracellular polymeric matrix formation, cyclic-di-GMP lifestyle switch from planktonic motility to sessile matrix attachment, and recalcitrant persister tolerance.",
        "Explain comparative respiratory physiology across insect tracheal direct gas phase diffusion, avian crosscurrent unidirectional parabronchial air sacs, fish countercurrent gill lamellae, and mammalian alveolar tidal pools.",
    ]

    math = [
        # Analysis, Topology, ODE/PDE (25)
        "Prove using epsilon-N and Bolzano-Weierstrass that every Cauchy sequence of real numbers converges in R, then construct a counterexample Cauchy sequence in Q failing to converge in Q.",
        "State and prove the uniform convergence theorem for interchange of limits and Riemann integrals, and exhibit a pointwise convergent sequence of continuous functions on [0,1] where int lim != lim int.",
        "Derive Taylor's theorem with Lagrange and integral forms of the remainder, bound |e^x - sum_{k=0}^n x^k/k!| on [-2, 2], and find minimum n for error < 10^-6.",
        "Prove the Contraction Mapping Theorem (Banach fixed-point theorem) on a complete metric space, quantify a priori error bounds, and apply it to show x = 0.5 cos(x) + 1 has a unique fixed point.",
        "Evaluate the contour integral int_{-inf}^{inf} dx / (x^4 + 1) using semicircular upper half-plane residue calculus, carefully justifying why the arc contribution vanishes.",
        "Use Cauchy's Argument Principle and Rouche's theorem to find the exact number of roots of z^5 + 4z^2 + 1 = 0 inside the annulus 1/2 < |z| < 2.",
        "Derive Laurent series expansions for f(z) = 1 / ((z - 1)(z - 3)) in all three concentric regions |z| < 1, 1 < |z| < 3, and |z| > 3, identifying singularities and residues.",
        "Solve the heat equation u_t = alpha u_xx on [0, L] with Dirichlet ends u(0,t)=u(L,t)=0 and initial profile u(x,0) = x(L-x) via separation of variables and Fourier sine coefficients.",
        "Derive D'Alembert's solution for the 1D wave equation u_tt = c^2 u_xx on (-inf, inf) with initial displacement f(x) and velocity g(x), and explain characteristic domains of dependence and influence.",
        "Use Frobenius method around the regular singular point x=0 to derive two linearly independent series solutions to x^2 y'' + x y' + (x^2 - 1/4) y = 0, expressing them in closed form.",
        "Apply Laplace transforms with convolution theorem to solve y'' + 4y' + 13y = e^(-t) sin(2t), y(0)=0, y'(0)=1, checking the final expression by differentiation.",
        "State Green's, Stokes', and Gauss Divergence theorems as special cases of generalized Stokes' theorem on manifolds, and compute surface flux of F=(x^3, y^3, z^3) out of sphere x^2+y^2+z^2=R^2.",
        "Prove Arzela-Ascoli compactness characterization in C([a,b]) via equicontinuity and uniform boundedness, illustrating how equicontinuity fails for sin(n x).",
        "Contrast Riemann integration with Lebesgue integration on characteristic function of rationals, and state Monotone Convergence, Fatou's Lemma, and Dominated Convergence Theorem.",
        "Perform phase-plane stability classification of equilibrium points for the damped nonlinear pendulum theta'' + b theta' + sin(theta) = 0 across b > 0, identifying saddles, spirals, and Lyapunov function.",
        "Solve the first-order quasilinear PDE u_t + u u_x = 0 (inviscid Burgers equation) with initial step data using method of characteristics, deriving shock formation time and Rankine-Hugoniot speed.",
        "Derive Euler-Lagrange equations from first variation of functional J[y] = int_a^b F(x, y, y') dx, and solve the brachistochrone cycloid problem and minimum surface of revolution catenoid.",
        "Apply Parseval's identity to evaluate sum_{n=1}^{inf} 1/n^4 from the Fourier series of f(x) = x^2 on [-pi, pi], verifying convergence step by step.",
        "Explain Heine-Borel compactness in R^n, construct an explicit open cover of (0,1) with no finite subcover, and show continuous images of compact sets attain maxima.",
        "Classify second-order linear PDEs (elliptic, parabolic, hyperbolic) via discriminant B^2 - 4AC, reduce u_xx + 2u_xy - 3u_yy = 0 to canonical coordinates, and solve its general form.",
        "Use the Sturm-Liouville self-adjoint eigenvalue framework to prove orthogonality of eigenfunctions and reality of eigenvalues for (p(x) y')' + q(x) y + lambda w(x) y = 0.",
        "Prove Jensen's inequality for convex functions via supporting hyperplanes, and deduce AM-GM and Holder's inequality for conjugate exponents 1/p + 1/q = 1.",
        "Compute the radius and exact interval of convergence (checking both endpoints) of sum_{n=1}^{inf} (-1)^n (2x - 3)^n / (n 5^n).",
        "Analyze bifurcation behavior as parameter r crosses 0 in the pitchfork x' = r x - x^3 and saddle-node x' = r + x^2 dynamical systems.",
        "Prove Baire Category Theorem for complete metric spaces and explain one consequence (existence of nowhere-differentiable continuous functions or Uniform Boundedness Principle).",
        # Linear Algebra, Algebra, Number Theory, Combinatorics (25)
        "Prove the Rank-Nullity Theorem dim(V) = rank(T) + nullity(T) by extending a kernel basis, and compute explicit bases for all four fundamental subspaces of [[1,2,0,3],[2,4,1,7],[1,2,1,4]].",
        "Compute the Jordan Canonical Form and generalized eigenvector transition basis P for 4x4 matrix A = [[2,1,0,0],[0,2,1,0],[0,0,2,0],[0,0,0,3]], then compute exp(At) explicitly.",
        "Prove Cayley-Hamilton theorem cleanly for diagonalizable matrices and extend over C by density, then apply it to express A^-1 for a 3x3 invertible matrix as a quadratic polynomial in A.",
        "Apply Gram-Schmidt orthogonalization to vectors v1=(1,1,0,1), v2=(1,-1,2,1), v3=(0,2,1,0) to construct an explicit QR factorization Q R.",
        "Prove the Spectral Theorem for real symmetric matrices (distinct eigenvalues -> orthogonal eigenvectors, induction on invariant orthogonal complement) and Sylvester's Law of Inertia for quadratic forms.",
        "Classify positive definite, semidefinite, and indefinite symmetric matrices via Sylvester's leading principal minors test, Cholesky A = L L^T existence, and eigenvalue signs.",
        "Prove Lagrange's theorem for finite groups via left coset partitioning, deduce Fermat's Little Theorem and Euler's totient theorem a^phi(n) ≡ 1 (mod n), and compute 7^2027 mod 100.",
        "State the First, Second, and Third Isomorphism Theorems for groups, and determine all groups of order 15 and order 21 up to isomorphism using Sylow theorems.",
        "Compare Principal Ideal Domains (PIDs), Unique Factorization Domains (UFDs), and Euclidean Domains, proving Z[sqrt(-5)] fails unique factorization via 6 = 2*3 = (1+sqrt(-5))(1-sqrt(-5)).",
        "Construct finite field GF(2^3) = F2[x]/(x^3 + x + 1), list all nonzero elements as powers of generator alpha, verify multiplicative inverses, and prove minimal polynomial properties.",
        "Derive Burnside's orbit-counting lemma and Polya enumeration theorem, counting distinct 2-colorings of vertices of a square under dihedral D4 symmetry and of a cube's faces under rotation.",
        "Solve the linear homogeneous recurrence a_n = 5 a_{n-1} - 8 a_{n-2} + 4 a_{n-3} with a_0=1, a_1=2, a_2=5 via characteristic roots and ordinary generating functions.",
        "Use exponential generating functions to prove the derangement count D_n = n! sum_{k=0}^n (-1)^k / k!, compute D_5, and derive Stirling numbers of the second kind inclusion-exclusion formula.",
        "Explain Catalan number C_n = (1/(n+1)) binom(2n,n) via reflection principle lattice paths, balanced parentheses, full binary trees, and derive its closed-form generating function.",
        "Prove the Euclidean algorithm produces gcd(a,b) in O(log min(a,b)) steps using Fibonacci worst-case bounds, solve 84x + 33y = 3, and find 37^-1 mod 1000.",
        "Explain Quadratic Reciprocity Law with Legendre symbols, determine whether x^2 ≡ 5 (mod 101) and x^2 ≡ 19 (mod 97) have solutions, and outline Tonelli-Shanks square roots mod p.",
        "Prove infinitely many primes of form 4k+3 exist via Euclid-style construction, contrast with 4k+1 sum-of-two-squares Fermat criterion, and verify Gaussian prime factorization.",
        "Prove Hall's Marriage Theorem for bipartite graph matchings using alternating paths or Max-Flow Min-Cut, and derive Konig's theorem equating maximum matching to minimum vertex cover.",
        "Prove Euler's planar graph formula V - E + F = 2 by induction on edges, deduce E <= 3V - 6 (and E <= 2V - 4 bipartite), and prove K5 and K3,3 are non-planar.",
        "Explain Ramsey's theorem R(3,3) = 6 proof via pigeonhole monochromatic triangles, upper-bound R(r,s) <= R(r-1,s) + R(r,s-1), and Erdos's probabilistic lower bound for R(k,k).",
        "Derive Mobius inversion on posets/integers sum_{d|n} mu(d) = [n=1] and apply it to count primitive binary strings of length n and irreducible polynomials of degree d over F_p.",
        "Explain Perron-Frobenius theorem for nonnegative irreducible matrices (positive leading eigenvalue, unique positive eigenvector, Google PageRank Markov convergence rate).",
        "Prove Dilworth's theorem partitioning finite posets into chains vs maximal antichain width, and relate it to Erdos-Szekeres monotone subsequence theorem of length nm+1.",
        "Explain RSA cryptosystem algebraic correctness for all messages m in Z_n (including gcd(m,n)>1 via CRT), Diffie-Hellman discrete-log exchange, and small-exponent Hastad broadcast attack.",
        "Construct and verify the [7,4,3] binary Hamming error-correcting code parity-check matrix H, syndrome decoding of single-bit errors, and Singleton/Hamming packing bounds.",
        # Probability, Statistics, Causal, Optimization, Numerical Methods (50 -> total 100)
        "Prove Boole's union bound, Markov's inequality, Chebyshev's variance bound, and Chernoff exponential tail bound for sum of independent Bernoulli random variables.",
        "Derive the joint distribution of order statistics X_(1) <= ... <= X_(n) for i.i.d. Uniform(0,1), prove X_(k) ~ Beta(k, n-k+1), and compute E[max - min].",
        "Given X, Y ~ i.i.d. Exponential(lambda), derive the distributions of U = X + Y and V = X / (X + Y) using Jacobian determinant transformation and prove U and V are independent.",
        "Explain bivariate normal distribution conditioning: derive E[Y|X=x], Var(Y|X=x), why zero covariance implies independence only for jointly normal RVs (giving uncorrelated dependent counterexample).",
        "State and prove the Gambler's Ruin absorbing Markov chain probability and expected stopping time using martingale optional stopping on symmetric and biased random walks.",
        "Derive M/M/1 and M/M/c Erlang-C continuous-time queueing steady-state distributions, Little's Law L = lambda W, and utilization rho -> 1 divergence behavior.",
        "Explain Poisson point processes: superposition, independent thinning, spatial Voronoi/volume property, and inspection paradox showing expected waiting interval is E[X^2]/E[X] rather than E[X]/2.",
        "Construct a branching Galton-Watson probability generating function G(s) recurrence, solve for extinction probability q as smallest fixed point in [0,1], and analyze criticality mu <= 1 vs mu > 1.",
        "Explain Brownian motion scaling, quadratic variation sum (dB)^2 -> t, Ito's lemma derivation for d f(t, B_t), and solving Geometric Brownian Motion SDE dS = mu S dt + sigma S dB.",
        "Derive Neyman-Pearson fundamental lemma for most powerful simple hypothesis tests, Wilks likelihood-ratio chi-square asymptotic theorem, and Type I/II error power calculations.",
        "Prove Gauss-Markov theorem that Ordinary Least Squares (OLS) is BLUE under homoscedastic uncorrelated zero-mean errors, and compute ridge regression bias-variance tradeoff.",
        "Derive the Cramer-Rao Lower Bound (CRLB) from score function variance Fisher information I(theta), check attainment in exponential families, and show Rao-Blackwell variance reduction.",
        "Explain sufficient statistics via Fisher-Neyman Factorization theorem, minimal sufficiency, completeness, and Lehmann-Scheffe theorem yielding UMVUE for Poisson and Normal families.",
        "Derive the Expectation-Maximization (EM) algorithm ELBO lower-bound ascent proof and write complete E-step and M-step update equations for a 1D k-component Gaussian Mixture Model.",
        "Explain Bayesian conjugate prior pairs (Beta-Binomial, Gamma-Poisson, Dirichlet-Multinomial, Normal-Normal), posterior predictive distributions, and Jeffreys invariant priors.",
        "Compare Metropolis-Hastings detailed balance acceptance probability, Gibbs sampling conditional sweeps, and Hamiltonian Monte Carlo (HMC) leapfrog symplectic energy preservation.",
        "Explain Bootstrap (nonparametric percentile, BCa, studentized pivotal) vs permutation randomization tests vs jackknife bias estimation, clarifying when bootstrap fails (e.g., max of Uniform).",
        "Contrast Pearl back-door adjustment, front-door criterion, instrumental variables (2SLS LATE estimand), propensity score IPW weighting, and difference-in-differences parallel trends assumptions.",
        "Explain Kaplan-Meier product-limit survival curve estimator with right-censoring, log-rank chi-square group test, and Cox proportional hazards partial likelihood inference.",
        "Compare multiple-testing control: Bonferroni/Holm family-wise error rate (FWER) versus Benjamini-Hochberg False Discovery Rate (FDR) step-up procedure at q=0.05.",
        "Explain Singular Value / QR / Cholesky / LU numerical linear algebra decompositions, floating-point condition number kappa(A), backward stability, and why normal equations square kappa(A).",
        "Derive Conjugate Gradient algorithm for symmetric positive-definite linear systems Ax=b via A-orthogonality of residuals and exact n-step termination without roundoff.",
        "Compare Power iteration, Inverse shifted iteration, and QR Francis shifted double-step eigenvalue algorithms, including deflation and convergence ratio |lambda_2/lambda_1|.",
        "Derive Newton-Raphson root-finding quadratic convergence condition, secant superlinear phi = 1.618 order, Broyden quasi-Newton updates, and safeguarded Brent's method.",
        "Compare Trapezoidal, Simpson's 1/3, Romberg Richardson extrapolation, and n-point Gauss-Legendre quadrature (exact up to degree 2n-1 using Legendre orthogonal polynomial roots).",
        "Analyze explicit Euler, implicit Backward Euler, Crank-Nicolson A-stability, and classic RK4 Butcher tableau stability regions on stiff ODE test equation y' = lambda y.",
        "Prove weak and strong duality in Linear Programming, Farkas' Lemma geometric interpretation, complementary slackness KKT conditions, and simplex Bland anti-cycling rule.",
        "Formulate Convex Optimization KKT optimality conditions (primal/dual feasibility, complementary slackness, stationarity), Slater constraint qualification, and Lagrangian dual of L1-regularized Lasso.",
        "Derive dual of Soft-Margin Support Vector Machine (SVM) with hinge loss, Mercer positive-semidefinite kernel trick, and geometric margin interpretation of support vectors.",
        "Analyze Newton's method vs BFGS/L-BFGS quasi-Newton secant curvature condition, backtracking Armijo line search, and proximal gradient FISTA O(1/k^2) composite acceleration.",
        "Prove convergence rate O(1/sqrt(T)) for projected stochastic subgradient descent on Lipschitz convex losses and Polyak-Ruppert averaging.",
        "Explain Multi-Armed Bandits: UCB1 Hoeffding optimism regret bound O(K ln T / Delta), Thompson Sampling Bayesian posterior matching, and EXP3 adversarial non-stochastic guarantees.",
        "Compare Value Iteration, Policy Iteration contraction mapping guarantees (Bellman operator gamma-contraction), Q-learning off-policy Robbins-Monro stochastic approximation, and TD(lambda) eligibility traces.",
        "Derive Kalman filter discrete-time linear-Gaussian predict/update Riccati covariance recursion and contrast with Extended Kalman Filter linearization vs Particle Filter SMC degeneracy.",
        "Explain Shannons Noisy-Channel Coding Theorem, mutual information I(X;Y) = H(Y) - H(Y|X), data processing inequality, and exact capacity C = W log2(1 + SNR) for AWGN channels.",
        "Prove Kraft-McMillan inequality for uniquely decodable codes, KL divergence Gibbs positivity D_KL(P||Q) >= 0, and Pinsker's inequality bounding Total Variation distance.",
        "Explain principal component analysis (PCA) maximum-variance / minimum-reconstruction equivalence, whitening ZCA/PCA transforms, and Johnson-Lindenstrauss random Gaussian projection lemma.",
        "Compare t-SNE Student-t crowding fix vs UMAP topological fuzzy simplicial cross-entropy, highlighting why global cluster distances and densities cannot be read directly from plots.",
        "Derive AdaBoost exponential loss greedy stagewise minimization: alpha_t = 0.5 ln((1-e_t)/e_t) and training error exponential decay when weak learners beat random guessing.",
        "Compare Random Forest bagging out-of-bag variance reduction vs Gradient Boosted Decision Trees second-order Newton Taylor leaves (XGBoost gain and regularization).",
        "Explain Shapley value game-theoretic axioms (efficiency, symmetry, dummy, additivity) in SHAP feature attribution versus permutation importance pitfalls with correlated predictors.",
        "Derive bias-variance decomposition for squared error, expected optimism in AIC/BIC/Mallows Cp, and why double descent appears at interpolation threshold p = n.",
        "Explain VC dimension Sauer-Shelah growth lemma, Rademacher complexity generalization bounds, and why L2 norm / flat margins explain overparameterized generalization.",
        "Compare spline interpolation natural cubic tridiagonal boundary conditions against Chebyshev polynomial node distribution minimizing Runge phenomenon on [-1,1].",
        "Derive fast polynomial multiplication via Number Theoretic Transform (NTT) modulo prime p = k 2^m + 1 using primitive roots of unity without floating-point rounding error.",
        "Explain spectral Chebyshev collocation vs finite difference vs Galerkin Finite Element Method (FEM) weak formulation with Lax-Milgram coercivity guarantee.",
        "Compare Monte Carlo variance reduction techniques: antithetic variates, control variates optimal beta, stratified sampling, importance sampling tilted likelihood ratio, and quasi-Monte Carlo Halton/Sobol.",
        "Explain Martingale convergence theorem, Azuma-Hoeffding bounded-differences concentration inequality, and Doob's maximal submartingale inequality.",
        "Derive Blackwell-Rao vs Stein's shrinkage estimator paradox showing James-Stein dominates sample mean under quadratic loss when dimension p >= 3.",
        "Compare Generalized Linear Models (GLMs): canonical link functions (logit, log, inverse), iteratively reweighted least squares (IRLS) Fisher scoring, deviance residuals, and overdispersion quasi-likelihood.",
    ]

    code = [
        # Python Algorithms, Data Structures, Async & Engineering (45)
        "Write clean Python implementing Aho-Corasick multi-pattern string automaton with failure links and dictionary-suffix output links in O(sum|patterns| + |text| + matches), plus tests.",
        "Implement a Suffix Automaton (SAM) in Python supporting O(N) construction, distinct-substring counting, longest common substring of two strings, and unit tests comparing against brute force.",
        "Write Python code implementing Z-algorithm and Knuth-Morris-Pratt (KMP) failure function, explaining prefix-border equivalence and adding property tests on random strings.",
        "Implement a heavy-light decomposition (HLD) + Euler-tour segment tree in Python answering path-max queries and subtree-add updates on weighted trees in O(log^2 N), with brute-force tests.",
        "Write clean Python implementing Edmonds-Karp and Dinic's maximum flow algorithm (level graph BFS + current-arc DFS blocking flow), testing on bipartite matching and min-cut graphs.",
        "Implement a persistent (immutable, path-copying) segment tree in Python answering k-th smallest element on arbitrary array subranges [l..r] in O(log N) per query.",
        "Write Python code for Johnson's all-pairs shortest paths (Bellman-Ford vertex reweighting + sparse Dijkstra) handling negative edge weights and reporting negative cycles cleanly.",
        "Implement an intrusive-free union-find Disjoint Set Union (DSU) with union by size, path halving, AND an undoable rollback-DSU for offline dynamic connectivity.",
        "Write a Python Treap (randomized BST by priority + key) supporting split, merge, k-th order statistic, and implicit-array range reversal with verified invariants.",
        "Implement Hungarian / Kuhn-Munkres O(N^3) minimum-weight bipartite assignment algorithm in Python with clear potential updates and differential check against itertools permutations for N<=8.",
        "Write Python implementing a Skip List with probabilistic levels supporting O(log N) expected search, insert, delete, and range scan, plus seed-reproducible tests.",
        "Implement a Fenwick (Binary Indexed) Tree supporting range-add and range-sum queries using two auxiliary BIT arrays in O(log N), explaining the algebraic prefix-sum derivation.",
        "Write a Python Manacher's O(N) longest palindromic substring solver handling odd and even centers uniformly, plus tests against O(N^2) center expansion.",
        "Implement topological sort with Kahn's lexicographically-smallest priority queue plus cycle reconstruction (returning exact cycle vertex list if DAG check fails).",
        "Write Python implementing 2-SAT solver via Kosaraju/Tarjan implication graph SCCs, returning a valid boolean assignment or proving unsatisfiability, with unit tests.",
        "Implement a streaming Count-Min Sketch + heavy-hitter min-heap tracker in Python with configurable epsilon/delta error bounds and empirical frequency-accuracy test.",
        "Write a Python context-free recursive-descent + Pratt (top-down operator precedence) arithmetic/boolean expression AST parser and evaluator supporting unary ops, exponentiation right-associativity, and variables.",
        "Implement a regex engine in Python compiling concatenation, alternation |, Kleene *, and ? to Thompson NFA states and running linear-time parallel state-set simulation (avoiding catastrophic backtracking).",
        "Write an external multi-way k-merge sort in Python operating on bounded in-memory chunks spilled to temporary files with tempfile cleanup and heapq loser/tournament merge.",
        "Implement a clean Python asyncio bounded worker pool with graceful shutdown cancellation, per-task timeouts, backpressure queue maxsize, and structured exception propagation.",
        "Write a zero-dependency Python JSON Lines WAL append-only key-value store with CRC32 record framing, crash-truncation recovery on load, and log compaction.",
        "Implement an async Python single-flight deduplication cache (coalescing concurrent identical cache-miss fetches so upstream runs once) handling exceptions and timeouts safely.",
        "Write a Python Merkle tree implementation with inclusion proof generation and independent verification (fixing second-preimage resistance via 0x00 leaf / 0x01 node domain prefixes).",
        "Implement a deterministic seeded Discrete-Event Simulator in Python using heapq for multi-server queue experiments, reporting p50/p95/p99 latency and utilization.",
        "Write clean Python implementing convex hull (Andrew's monotone chain O(N log N)), rotating calipers diameter/minimum-width pair, and closest pair of points divide-and-conquer.",
        "Implement a clean Python kd-tree for 2D/kD points supporting exact nearest-neighbor and k-nearest-neighbor backtracking pruning, verified against brute-force Euclidean scan.",
        "Write Python code solving Traveling Salesperson Problem bitmask DP O(n^2 2^n) reconstructing the optimal tour, alongside Held-Karp edge-case tests.",
        "Implement a trie-backed autocomplete engine in Python maintaining top-k weighted suggestions per prefix node with incremental frequency updates.",
        "Write a Python Myers O(ND) edit-script diff algorithm computing minimal line insertions/deletions with clear snake-path backtracking and roundtrip patch verification.",
        "Implement an immutable Persistent Vector / Rope data structure or balanced B-tree (order t=4) in Python supporting insert, split, search, and in-order range iteration.",
        "Write Python implementing a sliding-window monotonic deque maximum/minimum and earliest-subarray-sum >= K with negative elements (monotonic convex-hull prefix trick).",
        "Explain Python GIL evolution (3.13+ free-threaded PEP 703 biased reference counting, immortal objects), CPython bytecode specialization adaptive interpreter, and weakref cycle GC.",
        "Explain Python descriptors (__get__, __set__, __set_name__), data vs non-data precedence over __dict__, MRO C3 linearization, and __slots__ memory layout trade-offs.",
        "Compare Python multiprocessing fork vs spawn vs forkserver, shared_memory zero-copy NumPy IPC, asyncio epoll/kqueue selector loops, and structured concurrency TaskGroup.",
        "Implement a Python decorator library providing exponential backoff with full jitter, circuit breaker half-open recovery state machine, and mock-clock unit tests.",
        "Write a Python linear-algebraic Gaussian elimination solver over GF(2) and over Q (using fractions.Fraction) returning particular solution + nullspace basis.",
        "Implement a clean Python SAT DPLL + two-watched-literals / unit-propagation solver handling CNF clauses, proving soundness on pigeonhole & random 3-SAT instances.",
        "Write clean Python implementing Wavefront / Dijkstra grid multi-source BFS + 0-1 BFS deque shortest path and verify equivalence against Bellman-Ford.",
        "Implement an interval tree / sweep-line rectangle union area calculator in O(N log N) using coordinate-compressed segment tree covering counts in Python.",
        "Write a Python deterministic finite-state machine YAML/JSON validator or finite-difference sparse CSR matrix-vector multiply + Jacobi preconditioner CG solver.",
        "Implement a clean Python LZW or LZ77 sliding-window compressor + decompressor verifying exact round-trip bit-identity on binary and repetitive text buffers.",
        "Write a Python lock-free-style copy-on-write snapshot MVCC in-memory key-value map supporting begin_tx, read-at-snapshot, write-set conflict detection, and abort/commit.",
        "Implement a topological critical-path PERT/CPM scheduler in Python computing earliest/latest start times, slack, and critical-path edges for arbitrary DAG dependencies.",
        "Write Python implementing Pollard's rho integer factorization + Miller-Rabin deterministic primality test for 64-bit integers with exact bases.",
        "Explain Python typing covariance/contravariance (TypeVar, ParamSpec, Protocol structural subtyping, Generic variance rules) with concrete safe vs unsafe Liskov examples.",
        # C++20, Rust, Systems, Compilers, Databases, GPU/Networking (75 -> total 120)
        "Write modern C++20 implementing a custom arena / bump allocator compatible with std::pmr::memory_resource, alignment handling, reset semantics, and intrusive free-list pooling.",
        "Implement a generic C++20 small-buffer-optimized (SBO) type-erased move-only function wrapper (`MoveOnlyFn`) avoiding heap allocation for callables <= 32 bytes.",
        "Write C++20 thread-safe object pool + work-stealing Chase-Lev deque sketch using std::atomic explicit memory orderings (`relaxed`, `acquire`, `release`, `seq_cst`) explaining the ABA problem.",
        "Implement a C++20 constexpr compile-time string-hashing switch / trie lookup table and static_assert unit checks using consteval and std::string_view.",
        "Write clean C++20 implementing a hash map with open addressing and Robin Hood linear probing, tombstone-free backward-shift deletion, and load-factor resizing.",
        "Implement a C++20 coroutine `Generator<T>` and lazy `Task<T>` awaitable type illustrating symmetric transfer to prevent stack overflow on chained co_await calls.",
        "Explain C++ value categories (lvalue, xvalue, prvalue), guaranteed copy elision, std::move vs std::forward reference collapsing, and strict aliasing `-fstrict-aliasing` UB traps + std::bit_cast.",
        "Compare Rust ownership/borrow-checker lifetimes, Interior Mutability (`Cell`, `RefCell`, `Mutex`, `RwLock`), Pin/Unpin self-referential async Futures, and `Send` vs `Sync` auto-traits.",
        "Write idiomatic Rust implementing an arena-backed directed graph or trie without Rc<RefCell<>> using generational indices (`slotmap` pattern) and safe iteration.",
        "Implement an epoll/kqueue non-blocking TCP echo + HTTP framing state machine sketch in Rust or C++ explaining edge-triggered EAGAIN drain loops and write-readiness backpressure.",
        "Design a Log-Structured Merge-tree (LSM) storage engine: memtable skip list, immutable SSTable compaction, leveled vs size-tiered write/space/read amplification, Bloom filters, and WAL.",
        "Compare B-tree page latch crabbing (optimistic lock coupling) against Bw-tree lock-free mapping tables and ARIES physiological redo/undo crash recovery with CLR log records.",
        "Analyze SQL query optimization: Volcano iterator vs vectorized column-at-a-time vs HyPer LLVM JIT tuple-at-a-time compilation, join order DP, and hash join grace partitioning.",
        "Write SQL (PostgreSQL dialect) queries using recursive CTEs for hierarchical path rollup, window functions (`ROWS BETWEEN` vs `RANGE BETWEEN`), gap-and-islands sessionization, and keyset pagination.",
        "Explain MVCC vacuuming in PostgreSQL (tuple xmin/xmax visibility snapshots, transaction ID wraparound freeze, HOT updates) versus MySQL InnoDB rollback-segment undo chains.",
        "Compare Two-Phase Commit (2PC) coordinator-blocking failure modes against Three-Phase Commit and Saga orchestration/choreography compensations for distributed transactions.",
        "Explain Lamport logical clocks, vector clocks causal conflict detection, Hybrid Logical Clocks (HLC), and TrueTime bounded-uncertainty Spanner external consistency.",
        "Design a consistent hashing ring with virtual nodes, bounded-load hashing, and Jump Consistent Hash, comparing key-movement fractions and memory footprint when nodes fail.",
        "Compare CRDTs (state-based CvRDT semilattice merge vs op-based CmRDT): G-Counter, PN-Counter, OR-Set (observed-remove tag tombstones), and LWW-Element-Set clock-skew caveats.",
        "Explain gossip epidemic protocols (SWIM failure detection with indirect ping-req, anti-entropy Merkle tree sync, phi accrual heartbeat detectors) for large clusters.",
        "Trace SSA (Static Single Assignment) compiler construction: dominance frontiers, iterated dominance phi-node placement, sparse conditional constant propagation (SCCP), and Chaitin graph-coloring register allocation.",
        "Compare trace JIT (LuaJIT hot loops), method JIT (V8 Maglev/Turbofan speculative inline caches + OSR deoptimization), and WebAssembly linear-memory sandboxing + WASI capability security.",
        "Design an incremental Pratt/LR(1) parser vs Earley parser vs GLR, comparing shift-reduce conflict resolution with PEG ordered choice packrat memoization.",
        "Explain linker & loader mechanics: ELF sections vs segments, Position-Independent Code (PIC), GOT/PLT lazy binding, ASLR relocation overhead, and LTO ThinLTO cross-module inlining.",
        "Describe OS page cache vs O_DIRECT I/O, io_uring submission/completion ring shared memory syscalls (SQPOLL, registered fixed buffers), and zero-copy sendfile/splice mechanics.",
        "Explain Linux CFS (Completely Fair Scheduler) / EEVDF virtual runtime red-black tree, priority inheritance preventing mutex priority inversion, and RCU (Read-Copy-Update) grace periods.",
        "Compare x86-64 TSO (store-buffer total store order) against ARMv8 weakly-ordered memory model (load/store reordering, DMB ISH fences, LDAPR/STLR release-acquire instructions).",
        "Explain CPU MESI cache coherence protocol, store buffer & invalidation queue motivations for memory barriers, false sharing on 64/128-byte cache lines, and hardware prefetchers.",
        "Describe branch prediction internals: BTB, two-level adaptive TAGE predictors, return address stack (RAS), Spectre v1/v2 speculative side-channel leakage, and cmov branchless tradeoffs.",
        "Write a Metal/CUDA parallel prefix-sum (Blelloch exclusive scan) kernel design explaining bank-conflict-free index padding (`OFFSET = idx + (idx >> 5)`), work-efficiency, and multi-block decoupling.",
        "Design a fused FlashAttention tiling forward pass: online numerically stable softmax (`m_i`, `l_i` rescaling across K/V SRAM tiles) keeping HBM memory IO O(N d) instead of O(N^2).",
        "Explain quantized GEMM/GEMV dequantization kernel optimization on Apple Silicon Metal / CUDA: threadgroup cooperative vs register tiling, FP8/INT4 bit-unpacking, SIMD shuffle reductions, and occupancy limits.",
        "Compare PagedAttention KV-cache virtual memory block tables against radix-tree prefix caching (SGLang RadixAttention) for copy-on-write multi-turn LLM sharing.",
        "Describe speculative decoding mathematical proof of lossless target distribution preservation for greedy and stochastic (rejection-sampling residual `max(0, p-q)`) drafts, plus tree attention masks.",
        "Explain TLS 1.3 1-RTT ECDHE handshake, X25519 + HKDF key schedule, AEAD ChaCha20-Poly1305 nonce discipline, 0-RTT replay vulnerability, and post-quantum ML-KEM hybrid key exchange.",
        "Compare HTTP/2 HPACK + single-TCP head-of-line stalling vs HTTP/3 QUIC independent stream multiplexing, connection ID migration, and QPACK out-of-order dynamic table recovery.",
        "Design a BGP internet routing system explanation covering AS-path loop prevention, iBGP route reflectors, RPKI prefix origin validation, and anycast ECMP flow pinning.",
        "Explain Linux eBPF verifier safety invariants (bounded loops, DAG CFG checks, pointer arithmetic tracking), XDP wire-speed packet drop, and kprobe latency histograms.",
        "Compare Docker/Linux namespaces (PID, NET, MNT, USER) and cgroups v2 memory pressure / CPU bandwidth throttling against microVM hardware isolation (Firecracker KVM).",
        "Design a distributed rate limiter across edge POPs comparing token bucket, leaky bucket, sliding-window log, and sliding-window counter approximations with Redis Lua atomic scripts.",
        "Write clean Rust or C++20 code implementing an LRU-K or W-TinyLFU cache admission policy (Count-Min Sketch frequency + Doorkeeper Bloom filter + Segmented LRU).",
        "Explain Reed-Solomon erasure coding over GF(2^8) Vandermonde/Cauchy matrices in distributed object stores comparing (10,4) erasure vs 3x replication durability and rebuild amplification.",
        "Trace garbage collection algorithms: tri-color mark-sweep incremental write barriers (Dijkstra vs Yuasa snapshot-at-the-beginning), generational card tables, and ZGC colored pointers.",
        "Explain floating-point IEEE-754 binary64/binary16/bfloat16/FP8-E4M3 encodings, subnormal gradual underflow, Kahan compensated summation, FMA single rounding, and catastrophic cancellation.",
        "Design a time-series database (TSDB) engine using Gorilla XOR delta-of-delta timestamp and float compression, inverted label bitmaps, and retention compaction.",
        "Compare protobuf / FlatBuffers / Cap'n Proto binary wire formats: varint zigzag encoding, backward/forward field evolution rules, zero-copy mmap access, and alignment pitfalls.",
        "Implement a clean Python or C++ parser/evaluator for Datalog bottom-up semi-naive fixed-point evaluation on transitive closure and reachability rules.",
        "Explain symmetric cryptography AES-GCM GHASH authentication catastrophic nonce-reuse key leakage versus AES-GCM-SIV nonce-misuse resistance.",
        "Compare zero-knowledge proof building blocks: Schnorr interactive Sigma protocol, Fiat-Shamir heuristic, Pedersen commitments hiding/binding, and R1CS arithmetic circuits.",
        "Design a high-throughput distributed message log (Kafka-style): sequential disk append + pagecache zero-copy sendfile, ISR high-watermark replication, and transactional idempotent producers.",
        "Write a clean C++20 SIMD-friendly (portable or NEON/AVX2 style) UTF-8 validator finite-state or bitmask verifier + scalar edge fallback.",
        "Explain database query plan anomaly causes: correlated predicates independence error, parameter sniffing plan regression, skew in hash partitioning, and adaptive join switching.",
        "Design an observability pipeline collecting OpenTelemetry distributed traces, adaptive tail-based sampling preserving error/high-latency traces, and cardinality explosion guards on metrics.",
        "Compare static vs dynamic type-checking algorithms: Hindley-Milner Algorithm W let-polymorphism occurs check, bidirectional type checking, and gradual typing blame tracking.",
        "Explain LLVM/Clang loop optimizations: loop-invariant code motion (LICM), induction variable scalar evolution (SCEV), polyhedral loop tiling/interchange, and SLP auto-vectorization.",
        "Write clean Python/Rust code implementing bounded SPSC lock-free ring buffer verification or Tarjan's bridge and articulation-point finding algorithm with unit tests.",
        "Design a read-heavy global feature-flag / configuration service with CDN edge caching, long-poll / SSE delta push, versioned ETag monotonic guarantees, and circuit-broken stale reads.",
        "Explain cryptographic hash constructions: Merkle-Damgard length-extension vulnerability, sponge Keccak/SHA-3 capacity vs rate security, and Blake3 Merkle-tree SIMD parallelism.",
        "Compare Linux systemd socket activation, cgroup OOM-killer scores, transparent hugepages (THP) khugepaged latency spikes, and NUMA first-touch page allocation faults.",
        "Write C++20 code implementing an intrusive red-black tree or B+ tree node split/search skeleton with invariant checker validating black-height and BST ordering.",
        "Explain formal verification model checking: LTL vs CTL temporal logic properties (safety vs liveness), symbolic BDD state traversal, and SAT-based bounded model checking.",
        "Compare WebRTC ICE STUN/TURN NAT traversal hole-punching, DTLS-SRTP media encryption, TWCC congestion feedback, and SFU simulcast/SVC layer switching.",
        "Design a deterministic simulation testing framework (FoundationDB style) abstracting network sockets, disks, clock jitter, and PRNG scheduling seeds to catch split-brain bugs.",
        "Explain C++/Rust build systems & ABI stability: header modules vs C++20 modules BMI, symbol mangling, vtable layout fragility, opaque PIMPL ABI firewall, and PGO/BOLT binary layout.",
        "Compare multi-tenant SaaS PostgreSQL isolation architectures: shared schema row-level security (RLS) vs schema-per-tenant vs database-per-tenant vs Silo-based cell routing.",
        "Explain DNS infrastructure deeply: recursive resolver caching, authoritative anycast, DNSSEC RRSIG/DS chain of trust, negative NSEC3 proof, and DoH/DoT privacy trade-offs.",
        "Design a geospatial H3 hexagonal / S2 Hilbert-curve spatial index supporting fast k-ring proximity queries, polygon covering, and balanced range-partitioned sharding.",
        "Explain ARM64 Apple Silicon microarchitecture features: wide 8-wide decode reorder buffer, unified CPU/GPU/NPU coherent SLC, AMX matrix coprocessor, and pointer authentication PAC.",
        "Write Python code implementing exact Ford-Fulkerson min-cost max-flow using SPFA / Bellman-Ford potentials + Dijkstra successive shortest augmenting paths.",
        "Compare compiler backend instruction selection (maximal munch DAG tiling, dynamic programming BURS) and list scheduling latency-hiding heuristics.",
        "Design an idempotent payment-ledger double-entry microservice schema in SQL preventing race double-spends via serializable/SELECT FOR UPDATE ordering and monotonic ledger invariants.",
        "Explain zero-trust mTLS SPIFFE/SPIRE workload identity attestation, short-lived X.509 SVID rotation, OAuth2 PKCE vs client credentials, and JWT EdDSA key rollover JWKS pitfalls.",
        "Compare Kubernetes scheduler predicate/scoring phases, kube-proxy iptables O(N) vs eBPF Cilium direct O(1) service routing, and PodDisruptionBudget eviction safety.",
        "Explain SMT solver DPLL(T) architecture combining CDCL SAT core with Nelson-Oppen theory combination for bitvectors, linear real arithmetic simplex, and uninterpreted functions.",
        "Write clean Python implementing parallel topological Kahn scheduler using ThreadPoolExecutor respecting DAG barriers and propagating upstream failure fast-abort cleanly.",
    ]

    tech = [
        # Physics, Optics, Fluids, E&M, Control (40)
        "Derive Euler-Lagrange equations for the double pendulum, identify cyclic coordinates and conserved Noether quantities when L is time- and translation-invariant, and contrast Hamiltonian phase-space Liouville volume preservation.",
        "Explain rigid-body Euler rotation equations, principal inertia tensor diagonalization, why intermediate-axis rotation is unstable (Tennis Racket / Dzhanibekov theorem), and gyroscope torque-induced precession.",
        "Derive Maxwell's equations in differential and integral forms, obtain the wave equation in linear dielectric media, prove Poynting energy flux conservation, and compute Fresnel Brewster angle polarization.",
        "Explain transmission line telegrapher equations: characteristic impedance Z_0 = sqrt(L/C), reflection coefficient Gamma, quarter-wave impedance transformer matching, and Smith chart standing-wave ratio.",
        "Derive relativistic Doppler shift, twin-paradox proper time integration, invariant energy-momentum relation E^2 = (pc)^2 + (m c^2)^2, and threshold kinetic energy for pion production.",
        "Explain quantum angular momentum ladder operators L_+ and L_-, Clebsch-Gordan addition of two spin-1/2 particles into singlet/triplet Bell states, and Stern-Gerlach sequential measurement.",
        "Apply first-order and second-order time-independent Rayleigh-Schrodinger perturbation theory to the Stark shift of n=2 hydrogen states (degenerate perturbation) and helium variational ground energy.",
        "Explain quantum tunneling WKB transmission coefficient exp(-2 int kappa dx), alpha-decay Geiger-Nuttall log-lifetime scaling, and scanning tunneling microscope exponential tip-height sensitivity.",
        "Derive blackbody Planck spectral radiance solving ultraviolet catastrophe, Stefan-Boltzmann T^4 scaling, Bose-Einstein condensation critical temperature T_c, and Fermi-Dirac Sommerfeld electron heat capacity.",
        "Explain Navier-Stokes momentum conservation, Reynolds number laminar-to-turbulent transition, Blasius boundary-layer thickness, Bernoulli assumptions, and why adverse pressure gradients trigger separation.",
        "Compare compressible subsonic versus supersonic nozzle area-velocity relations, normal Rankine-Hugoniot shock entropy jump, Prandtl-Meyer expansion fans, and Mach cone shock angles.",
        "Derive acoustic wave impedance matching, organ-pipe open vs closed harmonic standing resonances, Helmholtz cavity resonator frequency, and Doppler supersonic shock cones.",
        "Explain geometric vs wave optics: Fraunhofer single-slit sinc^2 diffraction vs N-slit grating resolving power, Michelson vs Fabry-Perot finesse, and thin-film antireflection destructive interference.",
        "Compare laser 3-level ruby vs 4-level Nd:YAG population inversion thresholds, Q-switching giant pulses vs mode-locking femtosecond frequency combs.",
        "Explain PID closed-loop feedback control stability: transfer function poles/zeros, Bode gain/phase margin, Nyquist encirclement criterion, integral windup anti-saturation, and derivative kick filtering.",
        "Compare Linear Quadratic Regulator (LQR) algebraic Riccati optimal state-feedback gain against Model Predictive Control (MPC) receding-horizon inequality constraint handling.",
        "Explain Kalman-Bucy observability and controllability Gramians via PBH rank tests, Luenberger observer pole placement duality, and separation principle in LQG.",
        "Derive magnetic vector potential A, Biot-Savart law, Faraday motional vs transformer EMF, skin-depth exponential attenuation delta = sqrt(2/(omega mu sigma)), and waveguide TE10 cutoff frequency.",
        "Explain Carnot, Otto, Diesel, Brayton, and Rankine thermodynamic cycles on P-V and T-S diagrams, regeneration/intercooling efficiency gains, and vapor-compression COP limits.",
        "Derive conductive 1D transient heat diffusion analytical solution vs convective Nusselt/Prandtl/Grashof dimensionless numbers and radiative gray-body view-factor networks.",
        "Compare analog BJT vs MOSFET small-signal hybrid-pi models, Cascode current-mirror output resistance boosting, Miller capacitance pole splitting, and CMOS subthreshold leakage.",
        "Explain sigma-delta analog-to-digital oversampling noise shaping, SAR binary-weighted capacitor DAC vs Flash comparator-ladder power scaling, and Nyquist anti-aliasing filters.",
        "Compare brushed DC, brushless BLDC trapezoidal commutation, and Permanent Magnet Synchronous Motor Field-Oriented Control (FOC Park/Clarke dq-frame vector transforms).",
        "Explain phase-locked loop (PLL) phase-frequency detector, charge-pump loop filter second-order dynamics, VCO phase-noise integration, and fractional-N spurs.",
        "Describe spread-spectrum modulation (DSSS CDMA, OFDM cyclic prefix multipath ISI immunity, LDPC/Polar belief propagation), QAM constellation Eb/N0 BER waterfall, and Shannon limit.",
        "Explain antenna array factor phased-steering beams, Friis free-space path-loss link budget, radar range R^-4 scaling, and synthetic aperture radar (SAR) azimuth compression.",
        "Compare piezoelectric accelerometer sensing, MEMS capacitive gyroscope Coriolis coupling, strain-gauge Wheatstone bridge temperature compensation, and Kalman IMU sensor fusion.",
        "Explain structural mechanics: Euler-Bernoulli beam bending M/I = E/R, Mohr's circle principal stresses, Tresca vs von Mises yield criteria, and Euler column buckling critical load.",
        "Compare satellite orbital mechanics: vis-viva equation, Hohmann vs bi-elliptic orbital transfer delta-v savings, J2 sun-synchronous nodal precession, and Lagrange L1-L5 stability.",
        "Explain nuclear reactor neutron six-factor formula, delayed-neutron control margin (beta_eff), Doppler fuel temperature negative feedback, and Xenon-135 iodine-pit poisoning transient.",
        "Compare tokamak magnetic confinement safety factor q, D-T fusion Lawson triple product n T tau_E ignition criteria, versus inertial confinement laser ablation compression.",
        "Explain crystal lattice reciprocal space Brillouin zones, phonon acoustic vs optical dispersion branches, Umklapp thermal resistance, and topological insulator edge conductivity.",
        "Describe superconductor BCS Cooper-pair phonon attraction, Type I Meissner critical field vs Type II Abrikosov flux-pinning vortices, and SQUID Josephson flux quantization.",
        "Explain photovoltaics Shockley-Queisser single-junction ~33% detailed-balance efficiency limit, fill factor parasitic series/shunt resistance losses, and tandem perovskite-silicon cells.",
        "Compare piezoelectric, magnetostrictive, and electrocaloric smart transducer physics, hysteresis Preisach modeling, and impedance-matching ultrasonic medical imaging.",
        "Explain acoustic room Reverberation Time Sabine/Eyring formulas, active noise cancellation adaptive LMS anti-phase convergence delays, and beamforming microphone arrays.",
        "Compare high-frequency RF impedance spectroscopy, vector network analyzer (VNA) 12-term SOLT error calibration, S-parameter passivity/reciprocity, and TDR fault location.",
        "Explain microfluidics low-Reynolds-number Stokes reversibility, Dean vortex passive spiral cell sorting, capillary electrokinetic zeta potential, and droplet T-junction breakup.",
        "Describe fiber-optic single-mode dispersion (chromatic material + waveguide), polarization-mode PMD, Erbium-doped fiber amplifier (EDFA) 1550 nm window, and coherent DP-16QAM.",
        "Explain astro-imaging adaptive optics Shack-Hartmann wavefront sensing, Kolmogorov atmospheric turbulence Fried parameter r_0, and interferometric aperture synthesis Van Cittert-Zernike theorem.",
        # ML Systems, Architecture Trade-offs & Troubleshooting Runbooks (40 -> total 80)
        "Explain Transformer multi-head self-attention, FlashAttention IO-awareness, Grouped-Query Attention (GQA) KV-cache memory savings, RoPE relative frequency extrapolation (YaRN/NTK), and SwiGLU FFN parameters.",
        "Compare Mixture-of-Experts (MoE) routing algorithms: top-k softmax vs sigmoid + auxiliary-loss-free bias balancing, shared experts, all-to-all token dispatch imbalance, and EP communication overlap.",
        "Explain linear/state-space hybrid attention models (Mamba-2, Gated DeltaNet / KDA): chunkwise WY parallel prefill O(L) vs constant O(1) recurrent decode state update, comparing recall trade-offs vs full attention.",
        "Contrast Pre-Norm (RMSNorm), Post-Norm, and DeepNorm stabilization in 70B+ transformers, AdamW decoupled weight decay memory footprint (bf16 weights + fp32 m/v/master states), and ZeRO-1/2/3 sharding.",
        "Explain LLM distributed training parallelism dimensions: Data Parallel all-reduce vs Tensor Parallel Megatron column/row linear split vs Pipeline 1F1B bubble overhead vs Context Parallel Ring Attention.",
        "Analyze post-training alignment methods: Supervised Fine-Tuning (SFT) vs PPO actor-critic reward-model KL penalty vs Direct Preference Optimization (DPO) implicit reward vs GRPO group-relative RL.",
        "Explain LoRA, QLoRA (NF4 double quantization + paged optimizers), and DoRA weight-decomposed directional updates: rank selection, target module coverage, and zero-overhead inference weight merging.",
        "Compare post-training quantization formats: INT4 AWQ activation-aware channel scaling, GPTQ second-order Hessian error compensation, MXFP4 shared E8M0 microscaling, and FP8 E4M3/E5M2 per-block scales.",
        "Explain continuous batching (iteration-level scheduling) with chunked prefill preventing TTFT stalls during long-context mixed decode, and compute-bound prefill vs memory-bandwidth-bound decode rooflines.",
        "Compare speculative decoding families: Medusa independent heads, EAGLE autoregressive hidden-state feature fusion + draft tree, Lookahead Jacobi parallel verification, and target-drafter alignment.",
        "Design an inference serving autoscaler & router architecture balancing prefix-cache KV affinity hashing against queue-depth least-loaded routing under mixed 512-tok and 64k-tok prompts.",
        "Compare vector ANN indexing algorithms: HNSW hierarchical navigable small-world graph vs IVF-PQ inverted Voronoi product quantization vs DiskANN Vamana SSD graph recall/memory curves.",
        "Explain hybrid RAG retrieval combining BM25 lexical sparse scoring, dense bi-encoder cosine embeddings via Reciprocal Rank Fusion (RRF), cross-encoder reranking, and chunking boundary pitfalls.",
        "Create a structured comparison table and evaluation checklist for choosing among SQLite WAL, PostgreSQL, DuckDB vectorized OLAP, and ClickHouse MergeTree across OLTP vs analytical workloads.",
        "Write an actionable incident troubleshooting runbook for diagnosing sudden p99 tail-latency spikes in a stateless API tier (covering CPU throttling, GC pauses, connection pool exhaustion, DNS timeouts, and downstream retry storms).",
        "Design a production canary deployment & progressive rollout architecture comparing blue-green cutover, automated error-budget rollback gates, shadow traffic mirroring, and schema expand-contract migrations.",
        "Explain diffusion probabilistic models (DDPM) score-matching objective equivalence, DDIM deterministic ODE trajectory sampling, classifier-free guidance (CFG), and flow matching rectified straight trajectories.",
        "Compare Vision Transformers (ViT patchify + MAE masked autoencoding), CLIP contrastive InfoNCE dual encoders, and modern object detectors (DETR Hungarian bipartite set loss vs YOLO anchors).",
        "Explain Graph Neural Networks: message-passing MPNN vs GAT multi-head attention, over-smoothing/over-squashing bottlenecks in deep GNNs, and 1-WL Weisfeiler-Leman expressiveness limits.",
        "Analyze neural network calibration and uncertainty: temperature scaling post-hoc NLL minimization, expected calibration error (ECE) reliability diagrams, deep ensembles, and conformal prediction coverage guarantees.",
        "Compare PyTorch `torch.compile` TorchDynamo bytecode graph capture, AOTAutograd backward fusion, Triton block-level SRAM codegen, and CUDA Graphs launch-overhead elimination.",
        "Design a real-time fraud-detection streaming ML architecture combining Kafka event ingestion, Flink stateful sliding aggregates, Redis online feature store, and offline-online training-serving parity checks.",
        "Explain automatic differentiation modes: forward-mode dual numbers JVP vs reverse-mode tape/graph VJP, activation checkpointing (rematerialization O(sqrt(N)) memory), and custom backward numerical gradient checks.",
        "Compare evaluation methodologies for code-generating and reasoning LLMs: Pass@k unbiased estimator, contamination detection, LLM-as-a-judge position/verbosity bias mitigation, and sandbox execution safety.",
        "Design a multi-region active-active cloud architecture covering DNS/Anycast health checks, read-local write-partitioned databases, conflict resolution, and chaos-engineering GameDay fault injection.",
        "Explain SRE observability pillars: RED/USE dashboards, golden signals (latency, traffic, errors, saturation), multi-window multi-burn-rate SLO alerting, and blameless postmortem 5-whys action items.",
        "Compare CI/CD artifact security pipelines: reproducible hermetic builds, SBOM generation (CycloneDX/SPDX), SLSA provenance signing, container distroless minimal attack surface, and secret rotation.",
        "Explain differential privacy (epsilon, delta)-DP sensitivity, Laplace vs Gaussian noise mechanisms, DP-SGD per-example gradient clipping + moments accountant, and privacy-utility trade-offs.",
        "Compare federated learning FedAvg client drift under non-IID data heterogeneity, scaffold control variates, secure aggregation masking, and communication compression top-k sparsification.",
        "Design a benchmark suite and regression gate plan for a storage/inference library: separating microbenchmarks from end-to-end throughput, statistical median/IQR variance bounds, and bit-exact correctness invariants.",
        "Compare WebGPU / Metal / Vulkan compute abstractions: command encoders, synchronization fences/barriers, bindless descriptor indexing, and unified vs discrete staging-buffer transfers.",
        "Write a systematic debugging checklist for diagnosing silent numerical NaN/Inf divergence during mixed-precision BF16/FP16 deep model training (activation scale overflow, epsilon underflow, LR warmup).",
        "Explain anomaly detection approaches in industrial time series: Isolation Forest path length, matrix profile exact subsequence discord, LSTM reconstruction vs seasonal STL decomposition residuals.",
        "Compare reinforcement learning policy optimization from REINFORCE baseline variance reduction, GAE lambda bias-variance tradeoff, Trust Region Policy Optimization (TRPO) Fisher conjugate gradient, to DPO.",
        "Explain audio/speech foundation pipelines: STFT log-Mel spectrograms, CTC blank-token alignment dynamic programming, Conformer convolution-augmented self-attention, and neural RVQ audio codecs.",
        "Design a high-throughput feature-store + offline point-in-time join preventing future target leakage in tabular fraud/credit pipelines using ASOF joins.",
        "Compare Linux cgroups v2 memory.high throttling vs OOM-kill, NUMA-aware worker pinning, jemalloc/mimalloc thread-local arenas vs glibc ptmalloc contention in multi-threaded servers.",
        "Explain cache invalidation strategies cleanly: cache-aside vs read/write-through vs write-behind, lease/versioned anti-stampede locking, negative caching, and CDC Debezium binlog invalidation.",
        "Design an edge-to-cloud OTA firmware rollout system ensuring A/B atomic partition fallback, Ed25519 signed delta patches, exponential rollback health windows, and thundering-herd staggering.",
        "Explain hardware-software co-design of systolic arrays (TPU MXU weight-stationary vs output-stationary dataflows), double-buffered SRAM DMA overlap, and sparsity 2:4 structured tensor cores.",
    ]

    assert len(med) == 140, f"med={len(med)}"
    assert len(chem) == 80, f"chem={len(chem)}"
    assert len(bio) == 80, f"bio={len(bio)}"
    assert len(math) == 100, f"math={len(math)}"
    assert len(code) == 120, f"code={len(code)}"
    assert len(tech) == 80, f"tech={len(tech)}"

    groups = [("med", med), ("chem", chem), ("bio", bio), ("math", math), ("code", code), ("tech", tech)]
    by_dom = {d: list(plist) for d, plist in groups}
    mixed = []
    i = 0
    while len(mixed) < 600:
        for d, _ in groups:
            if i < len(by_dom[d]):
                mixed.append((d, "train", by_dom[d][i]))
        i += 1
    assert len(mixed) == 600
    return [(idx, dom, split, text) for idx, (dom, split, text) in enumerate(mixed)]


def _word_ngrams(text: str, n: int = 6) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def check_no_leakage(items: list[tuple[int, str, str, str]]) -> None:
    from bench.exact import PROMPTS as EXACT_PROMPTS
    from tools.run_eval import PROMPTS as EVAL_PROMPTS

    orig_texts = {t.lower().strip() for _, _, _, t in _orig_prompts()}
    ref_texts = [p.lower().strip() for _, p in EXACT_PROMPTS] + [p.lower().strip() for _, p in EVAL_PROMPTS]
    ref_6grams = set()
    for p in ref_texts:
        ref_6grams.update(_word_ngrams(p, 6))

    all_forbidden = list(FORBIDDEN_SUBSTRINGS) + EXTRA_FORBIDDEN
    seen = set()
    for idx, dom, split, text in items:
        assert text and text.strip(), f"Empty prompt at idx={idx}"
        low = text.lower().strip()
        assert low not in seen, f"Duplicate prompt at idx={idx}: {text[:80]}"
        assert low not in orig_texts, f"Overlaps gen_dflash_data prompt at idx={idx}: {text[:80]}"
        seen.add(low)
        for ref in ref_texts:
            assert low != ref and ref[:40] not in low, f"Prompt {idx} overlaps eval prompt: {ref[:50]}"
        ng = _word_ngrams(low, 6)
        overlap = ng & ref_6grams
        assert not overlap, f"Prompt {idx} shares 6-gram {next(iter(overlap))} with eval prompt!"
        for bad in all_forbidden:
            if bad in ("stemi", "avf", "nsaid"):
                assert re.search(rf"\b{re.escape(bad)}\b", low) is None, (
                    f"Forbidden word {bad!r} in prompt {idx} ({dom}): {text[:80]}"
                )
            else:
                assert bad not in low, f"Forbidden substring {bad!r} in prompt {idx} ({dom}): {text[:80]}"

    print("Anti-leakage check PASS: 600 unique disjoint train prompts, 0 overlaps.", flush=True)


def run_rollouts(model_path: str, out_dir: Path, items: list, max_tokens: int, batch_size: int, save_every: int):
    import mlx_lm  # noqa: E402,F401
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.utils import maybe_set_recommended_wired_limit

    out_dir.mkdir(parents=True, exist_ok=True)
    ro_path = out_dir / "rollouts.json"
    done_by_idx: dict[int, dict] = {}
    if ro_path.exists():
        prev = json.loads(ro_path.read_text())
        for row in prev.get("rollouts", []):
            done_by_idx[int(row["idx"])] = row
        print(f"Resuming rollouts: {len(done_by_idx)}/{len(items)} already completed.", flush=True)

    todo = [it for it in items if it[0] not in done_by_idx]
    if not todo:
        print(f"All {len(items)} rollouts already generated.", flush=True)
        return

    maybe_set_recommended_wired_limit()
    t0 = time.perf_counter()
    model, tok = load(model_path)
    mx.eval(model.parameters())
    print(
        f"Loaded target for rollouts in {time.perf_counter()-t0:.1f}s "
        f"(fp8={getattr(model, '_lingsante_fp8', False)}, active={mx.get_active_memory()/1e9:.1f} GB)",
        flush=True,
    )

    eos_raw = getattr(tok, "eos_token_ids", None)
    if eos_raw:
        stop_ids = [[int(x)] for x in eos_raw]
    elif getattr(tok, "eos_token_id", None) is not None:
        stop_ids = [[int(tok.eos_token_id)]]
    else:
        stop_ids = []
    greedy = lambda lp: mx.argmax(lp, axis=-1)

    def fmt_ids(text: str) -> list[int]:
        s = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        return tok.encode(s, add_special_tokens=False)

    t_gen0 = time.perf_counter()
    for c_start in range(0, len(todo), save_every):
        chunk = todo[c_start : c_start + save_every]
        p_ids_list = [fmt_ids(text) for _, _, _, text in chunk]
        gen = BatchGenerator(
            model,
            max_tokens=max_tokens,
            sampler=greedy,
            completion_batch_size=batch_size,
            prefill_batch_size=1,
            stop_tokens=stop_ids,
        )
        uids = gen.insert(p_ids_list, max_tokens=[max_tokens] * len(chunk))
        uid_map = {u: k for k, u in enumerate(uids)}
        toks_acc = [[] for _ in chunk]
        fin_reasons = [None] * len(chunk)
        t_c0 = time.perf_counter()
        while True:
            p_resps, resps = gen.next()
            if not p_resps and not resps:
                break
            if not resps:
                continue
            for r in resps:
                k = uid_map[r.uid]
                toks_acc[k].append(int(r.token))
                if r.finish_reason is not None:
                    fin_reasons[k] = r.finish_reason
        gen.close()
        del gen
        mx.clear_cache()
        dt_c = time.perf_counter() - t_c0
        c_toks = sum(len(x) for x in toks_acc)

        for (idx, dom, split, _), p_ids, g_toks, fr in zip(chunk, p_ids_list, toks_acc, fin_reasons):
            done_by_idx[idx] = {
                "idx": idx,
                "domain": dom,
                "split": split,
                "prompt_len": len(p_ids),
                "gen_len": len(g_toks),
                "finish_reason": fr,
                "ids": p_ids + g_toks,
            }
        ordered = [done_by_idx[i] for i in sorted(done_by_idx)]
        ro_path.write_text(json.dumps({"model": model_path, "max_tokens": max_tokens, "rollouts": ordered}))
        avg_len = c_toks / len(chunk)
        print(
            f"  Rollouts {len(done_by_idx):3d}/{len(items)}: chunk +{len(chunk)} "
            f"({c_toks} tok, avg_gen={avg_len:.1f}, {c_toks/dt_c:.1f} tok/s, elapsed={time.perf_counter()-t_gen0:.0f}s)",
            flush=True,
        )

    all_rows = [done_by_idx[i] for i in sorted(done_by_idx)]
    short = [r["idx"] for r in all_rows if r["gen_len"] < 32]
    tot_t = sum(r["gen_len"] for r in all_rows)
    print(f"Rollouts complete: {len(all_rows)} seqs, total_gen_tok={tot_t}, mean_gen={tot_t/len(all_rows):.1f}, short(<32)={short}", flush=True)
    del model
    mx.clear_cache()


def run_capture(model_path: str, out_dir: Path, shard_size: int, prefill_step: int):
    import mlx_lm  # noqa: E402,F401
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.layers.attention import create_attention_mask
    from mlx_lm.layers.recurrent import create_ssm_mask
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import maybe_set_recommended_wired_limit

    ro_path = out_dir / "rollouts.json"
    assert ro_path.exists(), f"Missing {ro_path}; run --stage rollouts first"
    rollouts = json.loads(ro_path.read_text())["rollouts"]

    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    n_shards = (len(rollouts) + shard_size - 1) // shard_size
    all_done = all(
        (shard_dir / f"shard_{sid:03d}.safetensors").exists() and (shard_dir / f"shard_{sid:03d}.meta.json").exists()
        for sid in range(n_shards)
    )

    inner = None
    model = None
    if not all_done:
        maybe_set_recommended_wired_limit()
        t0 = time.perf_counter()
        model, _ = load(model_path)
        if hasattr(model, "_lingsante_mtp_holder"):
            model._lingsante_mtp_holder = None
        mx.clear_cache()
        mx.eval(model.parameters())
        print(f"Loaded target for capture in {time.perf_counter()-t0:.1f}s ({mx.get_active_memory()/1e9:.1f} GB)", flush=True)
        inner = model.model

        sio_path = out_dir / "shared_io.safetensors"
        if not sio_path.exists():
            w_emb = inner.word_embeddings.weight.astype(mx.bfloat16)
            w_head = model.lm_head.weight.astype(mx.bfloat16)
            mx.eval(w_emb, w_head)
            mx.save_safetensors(str(sio_path), {"word_embeddings.weight": w_emb, "lm_head.weight": w_head})
            print(f"Saved frozen I/O weights to {sio_path} ({sio_path.stat().st_size/1e9:.2f} GB)", flush=True)

    cap_set = set(TARGET_LAYER_IDS)
    index_entries = []
    tot_pos = 0
    tot_match = 0
    tot_top4 = 0
    t_cap0 = time.perf_counter()

    for sh_id, sh_start in enumerate(range(0, len(rollouts), shard_size)):
        batch_rows = rollouts[sh_start : sh_start + shard_size]
        sh_name = f"shard_{sh_id:03d}.safetensors"
        sh_path = shard_dir / sh_name
        meta_path = shard_dir / f"shard_{sh_id:03d}.meta.json"
        if sh_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            index_entries.extend(meta["entries"])
            tot_pos += int(meta["tot_pos"])
            tot_match += int(meta["tot_match"])
            tot_top4 += int(meta["tot_top4"])
            print(f"  Capture {sh_name}: skipped (already saved, {len(meta['entries'])} seqs)", flush=True)
            continue

        tensors = {}
        sh_entries = []
        sh_pos = 0
        sh_match = 0
        sh_top4 = 0
        sh_t0 = time.perf_counter()
        sh_toks = 0

        for r in batch_rows:
            idx = int(r["idx"])
            ids = r["ids"]
            plen = int(r["prompt_len"])
            T = len(ids)
            p_arr = mx.array(ids, dtype=mx.uint32)
            cache = make_prompt_cache(model)
            feat_chunks = []
            hf_chunks = []
            t32_id_chunks = []
            t32_lg_chunks = []

            pos = 0
            while pos < T:
                n = min(prefill_step, T - pos)
                chunk = p_arr[pos : pos + n][None]
                h = inner.word_embeddings(chunk)
                attn_mask = (
                    create_attention_mask(h, cache[inner.attention_index], return_array=True)
                    if inner.attention_index is not None
                    else None
                )
                lin_mask = (
                    create_ssm_mask(h, cache[inner.linear_index])
                    if inner.linear_index is not None
                    else None
                )
                captured = {}
                for li, (layer, lc) in enumerate(zip(inner.layers, cache)):
                    mask = lin_mask if layer.is_linear else attn_mask
                    h = layer(h, mask=mask, cache=lc)
                    if li in cap_set:
                        captured[li] = h
                feats_c = mx.concatenate([captured[li] for li in TARGET_LAYER_IDS], axis=-1)[0]
                hf_raw = inner.norm(h)
                logits_c = model.lm_head(hf_raw)[0]
                hf_c = hf_raw[0].astype(mx.bfloat16)
                lg_f32 = logits_c.astype(mx.float32)
                top_idx = mx.argpartition(-lg_f32, kth=31, axis=-1)[..., :32]
                top_vals = mx.take_along_axis(lg_f32, top_idx, axis=-1)
                ord_idx = mx.argsort(-top_vals, axis=-1)
                t32_ids_c = mx.take_along_axis(top_idx, ord_idx, axis=-1).astype(mx.uint32)
                t32_lgs_c = mx.take_along_axis(top_vals, ord_idx, axis=-1).astype(mx.float16)
                mx.eval(feats_c, hf_c, t32_ids_c, t32_lgs_c, [c.state for c in cache])
                feat_chunks.append(feats_c)
                hf_chunks.append(hf_c)
                t32_id_chunks.append(t32_ids_c)
                t32_lg_chunks.append(t32_lgs_c)
                pos += n

            feats_all = mx.concatenate(feat_chunks, axis=0) if len(feat_chunks) > 1 else feat_chunks[0]
            hf_all = mx.concatenate(hf_chunks, axis=0) if len(hf_chunks) > 1 else hf_chunks[0]
            t32_ids_all = mx.concatenate(t32_id_chunks, axis=0) if len(t32_id_chunks) > 1 else t32_id_chunks[0]
            t32_lgs_all = mx.concatenate(t32_lg_chunks, axis=0) if len(t32_lg_chunks) > 1 else t32_lg_chunks[0]
            mx.eval(feats_all, hf_all, t32_ids_all, t32_lgs_all)

            if T > plen + 1:
                t32_sub = t32_ids_all[plen - 1 : T - 1, :4].tolist()
                gold1 = ids[plen:T]
                n_cmp = len(gold1)
                n_hit = sum(1 for row, b in zip(t32_sub, gold1) if row[0] == b)
                n_top4 = sum(1 for row, b in zip(t32_sub, gold1) if b in row)
                rate = n_hit / n_cmp
                rate4 = n_top4 / n_cmp
                sh_pos += n_cmp
                sh_match += n_hit
                sh_top4 += n_top4
                if rate < 0.90 or rate4 < 0.99:
                    raise RuntimeError(f"Alignment mismatch on idx={idx}: top1={n_hit}/{n_cmp} ({rate:.3f}), top4={n_top4}/{n_cmp} ({rate4:.3f})")
            else:
                rate = 1.0
                rate4 = 1.0

            key = f"e{idx:03d}"
            tensors[f"{key}.ids"] = p_arr
            tensors[f"{key}.feats"] = feats_all.astype(mx.bfloat16)
            tensors[f"{key}.h_final"] = hf_all.astype(mx.bfloat16)
            tensors[f"{key}.top32_ids"] = t32_ids_all
            tensors[f"{key}.top32_logits"] = t32_lgs_all
            sh_entries.append({
                "idx": idx,
                "shard": sh_name,
                "key": key,
                "domain": r["domain"],
                "split": r["split"],
                "prompt_len": plen,
                "total_len": T,
                "align_rate": round(rate, 5),
                "top4_rate": round(rate4, 5),
            })
            sh_toks += T
            del cache

        mx.save_safetensors(str(sh_path), tensors)
        meta_path.write_text(json.dumps({"entries": sh_entries, "tot_pos": sh_pos, "tot_match": sh_match, "tot_top4": sh_top4}, indent=1))
        index_entries.extend(sh_entries)
        tot_pos += sh_pos
        tot_match += sh_match
        tot_top4 += sh_top4
        del tensors
        mx.clear_cache()
        sh_dt = time.perf_counter() - sh_t0
        print(
            f"  Capture {sh_name}: seqs={len(batch_rows)} ({sh_toks} tok, {sh_toks/sh_dt:.1f} tok/s), "
            f"top1={100.0*tot_match/max(1,tot_pos):.2f}%, top4={100.0*tot_top4/max(1,tot_pos):.2f}%, "
            f"elapsed={time.perf_counter()-t_cap0:.0f}s",
            flush=True,
        )

    global_rate = tot_match / max(1, tot_pos)
    global_rate4 = tot_top4 / max(1, tot_pos)
    print(
        f"GLOBAL ALIGNMENT SELF-TEST: top1={tot_match}/{tot_pos} ({100.0 * global_rate:.3f}%), "
        f"top4={tot_top4}/{tot_pos} ({100.0 * global_rate4:.3f}%)",
        flush=True,
    )
    assert global_rate >= 0.96, f"Global top1 rate {global_rate:.4f} < 0.96!"
    assert global_rate4 >= 0.995, f"Global top4 rate {global_rate4:.4f} < 0.995!"

    (out_dir / "index.json").write_text(
        json.dumps(
            {
                "model": model_path,
                "layer_ids": list(TARGET_LAYER_IDS),
                "has_inline_hfinal": True,
                "align_hits": tot_match,
                "align_top4_hits": tot_top4,
                "align_total": tot_pos,
                "align_rate": round(global_rate, 6),
                "top4_rate": round(global_rate4, 6),
                "entries": index_entries,
            },
            indent=1,
        )
    )
    print(f"Saved index.json ({len(index_entries)} sequences) in {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/science/models/lingsante-mlx-mxfp4-bf16"))
    ap.add_argument("--out-dir", default=os.path.expanduser("~/science/data/lingsante_eagle3_600"))
    ap.add_argument("--stage", default="all", choices=["check", "rollouts", "capture", "all"])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--shard-size", type=int, default=20)
    ap.add_argument("--prefill-step", type=int, default=512)
    a = ap.parse_args()

    if a.stage != "check":
        import mlx_lm  # noqa: E402,F401

    items = _build_prompts()
    check_no_leakage(items)
    if a.stage == "check":
        return 0
    out_dir = Path(a.out_dir).expanduser()
    if a.stage in ("rollouts", "all"):
        run_rollouts(a.model, out_dir, items, a.max_tokens, a.batch_size, a.save_every)
    if a.stage in ("capture", "all"):
        run_capture(a.model, out_dir, a.shard_size, a.prefill_step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
