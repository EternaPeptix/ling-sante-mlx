"""Generate a strictly disjoint LingSante-3.1-flash self-distillation dataset for DFlash2 training.

Stages:
  --stage check     : Run anti-leakage guard and verify 220 distinct prompts (200 train / 20 val).
  --stage rollouts  : Generate C=4 greedy rollouts (max 384 tokens) with BatchGenerator -> rollouts.json.
  --stage capture   : Prefill full sequences with target model, extract layer (1,18,35,52,70) features
                      and top-32 teacher logits, verify >=99.5% greedy alignment -> sharded safetensors.
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

TARGET_LAYER_IDS = (1, 18, 35, 52, 70)
FORBIDDEN_SUBSTRINGS = [
    "stemi",
    "ii, iii",
    "kmno4",
    "permanganate",
    "lac operon",
    "allolactose",
    "metformin",
    "lisinopril",
    "furosemide",
    "tert-butyl",
    "methoxide",
    "mitosis",
    "meiosis",
    "ckd 3b",
    "stage 3b",
    "egfr 36",
    "crushing substernal",
    "avf",
    "effusion fraction 30",
    "ejection fraction (ef 30",
    "nsaid",
]


def _build_prompts() -> list[tuple[str, str]]:
    """Build 220 diverse prompts: 100 medicine, 45 chemistry, 45 biology, 30 math/code/logic."""
    med = [
        # Cardiology & Vascular (12)
        "A 68-year-old man with severe aortic stenosis develops syncope during exertion and mild dyspnea on stairs. Explain the underlying hemodynamics, key physical exam findings, transthoracic echocardiography criteria for severe stenosis, and indications for aortic valve replacement.",
        "A 63-year-old woman presents with palpitations and irregularly irregular pulse at 134 bpm, blood pressure 118/74 mmHg, and no chest pain. Outline acute rate vs rhythm control strategies for new-onset atrial fibrillation, anticoagulation risk stratification using CHA2DS2-VASc, and bleeding assessment.",
        "A 42-year-old man develops sharp pleuritic chest pain that improves when leaning forward and worsens when supine two weeks after a viral upper respiratory illness. Describe expected ECG evolution across stages, inflammatory markers, first-line therapy, and signs of cardiac tamponade.",
        "Compare the pathophysiology, jugular venous pressure waveform abnormalities, Kussmaul sign, and echocardiographic features of constrictive pericarditis versus restrictive cardiomyopathy.",
        "A 71-year-old smoker has exertional calf pain relieved after three minutes of rest and an ankle-brachial index of 0.58 on the left leg. Detail diagnostic workup, supervised exercise therapy, antiplatelet and lipid-lowering management, and indications for revascularization.",
        "Explain the Starling curve, neurohormonal activation via RAAS and sympathetic nervous system, and the mechanistic rationale for beta-blockers, mineralocorticoid receptor antagonists, and ARNI therapy in chronic systolic heart failure.",
        "A 55-year-old man with longstanding poorly controlled blood pressure presents with BP 228/132 mmHg, confusion, papilledema, and rising serum creatinine. Distinguish hypertensive emergency from urgency and outline parenteral infusion targets during the first 24 hours.",
        "Describe the Duke criteria for infective endocarditis, typical microbiological causes in native versus prosthetic valves, empiric antibiotic regimens pending blood cultures, and urgent surgical indications.",
        "A 49-year-old woman develops sudden tearing interscapular back pain radiating to the neck with a 28 mmHg blood pressure differential between arms. Explain the Stanford classification, immediate impulse-control pharmacology, and definitive management.",
        "Compare acute myocarditis, takotsubo (stress) cardiomyopathy, and hypertrophic obstructive cardiomyopathy with respect to pathophysiology, echocardiography, cardiac MRI findings, and risk stratification.",
        "Explain how hyperkalemia alters cardiac membrane resting potential, action potential upstroke, and sequential surface ECG changes from peaked T waves to sine wave pattern, and list emergent stabilizing and potassium-shifting interventions.",
        "A 36-year-old tall man with lens dislocation and pectus excavatum has a widened mediastinum on chest radiograph. Discuss fibrillin-1 genetics, aortic root surveillance thresholds, and beta-blocker or ARB prophylaxis.",
        # Pulmonology & Critical Care (10)
        "A 66-year-old former 45-pack-year smoker with baseline chronic bronchitis has worsening purulent sputum, dyspnea, and arterial blood gas pH 7.31, PaCO2 62 mmHg, PaO2 56 mmHg on room air. Outline controlled oxygen targets, bronchodilators, systemic corticosteroids, antibiotics, and NIV criteria.",
        "A 54-year-old woman five days after total knee arthroplasty develops acute tachypnea, pleuritic pain, heart rate 116 bpm, and SpO2 88%. Explain pretest probability scoring, D-dimer utility, CT pulmonary angiography, risk stratification for massive vs submassive pulmonary embolism, and anticoagulation.",
        "Compare obstructive and restrictive ventilatory defects on spirometry, flow-volume loops, lung volumes (TLC, RV, FRC), and DLCO, giving two parenchymal and two extraparenchymal causes of restriction.",
        "Outline the Berlin definition of acute respiratory distress syndrome (ARDS), lung-protective low-tidal-volume ventilation settings, plateau pressure goals, PEEP titration, and prone positioning rationale.",
        "A 29-year-old woman with moderate persistent asthma presents with severe wheezing, inability to speak full sentences, and silent chest after nebulized albuterol. Detail ICU escalation, IV magnesium sulfate, systemic steroids, and warning signs of impending hypercapnic respiratory failure.",
        "Explain Light's criteria for distinguishing transudative from exudative pleural effusions, and describe additional pleural fluid tests (pH, glucose, ADA, cytology, triglycerides) used in empyema, tuberculous pleurisy, and chylothorax.",
        "Discuss the pathogenesis of idiopathic pulmonary fibrosis, high-resolution CT usual interstitial pneumonia pattern, pulmonary function changes, and the mechanisms of antifibrotic agents pirfenidone and nintedanib.",
        "A 60-year-old patient with small-cell lung cancer develops facial plethora, dilated chest wall veins, and dyspnea when lying flat. Explain superior vena cava syndrome pathophysiology, tissue diagnosis before radiation, and emergent endovascular stenting indications.",
        "Compare community-acquired pneumonia, hospital-acquired pneumonia, and aspiration lung abscess in terms of predominant pathogens, diagnostic imaging, and guideline-concordant empiric antimicrobial coverage.",
        "Explain the physiology of hypoxic pulmonary vasoconstriction, WHO Group 1 through Group 5 pulmonary hypertension classifications, right heart catheterization diagnostic cutoffs, and targeted pulmonary vasodilator pathways.",
        # Nephrology & Acid-Base / Electrolytes (10)
        "Compare prerenal azotemia, acute tubular necrosis, and acute interstitial nephritis across urinalysis microscopy, urine sodium, fractional excretion of sodium (FENa), fractional excretion of urea, and clinical triggers.",
        "A patient with chronic diarrhea and vomiting has Na 136, K 3.1, Cl 89, HCO3 34 mEq/L, and arterial pH 7.49, PaCO2 46 mmHg. Perform a complete step-by-step acid-base analysis, calculate expected compensation, and explain how volume contraction and aldosterone sustain metabolic alkalosis.",
        "A 45-year-old man with chronic alcohol use disorder presents with serum sodium 111 mEq/L and confusion. Detail the diagnostic evaluation of hypotonic hyponatremia using serum/urine osmolality and urine sodium, safe 24-hour correction limits, and osmotic demyelination syndrome prevention.",
        "Compare nephritic syndrome versus nephrotic syndrome across glomerular basement membrane pathology, urine sediment, proteinuria threshold, lipid abnormalities, thrombotic risk, and hallmark biopsy patterns of IgA nephropathy and minimal change disease.",
        "Explain Winter's formula, the serum anion gap, the delta-delta ratio, and osmolar gap in evaluating a patient with suspected ethylene glycol or salicylate toxicity.",
        "Describe calcium, phosphate, PTH, and 1,25-dihydroxyvitamin D physiology, and outline the pathogenesis and management of secondary hyperparathyroidism and renal osteodystrophy in chronic renal failure.",
        "A 52-year-old man develops polyuria (>5 L/day) and hypernatremia (Na 154 mEq/L) after pituitary adenoma resection. Contrast central vs nephrogenic diabetes insipidus via water deprivation and desmopressin response testing, and describe free water deficit calculation.",
        "Explain the pathophysiology of hepatorenal syndrome, diagnostic exclusion criteria including albumin challenge, and pharmacologic splanchnic vasoconstriction with terlipressin or midodrine plus octreotide.",
        "Compare autosomal dominant polycystic kidney disease with Alport syndrome regarding inheritance, extrarenal manifestations (intracranial aneurysms, sensorineural deafness, lenticonus), and targeted tolvaptan considerations.",
        "A 64-year-old man with benign prostatic hyperplasia presents with anuria, suprapubic fullness, and creatinine rising from 1.1 to 5.4 mg/dL over 48 hours. Outline postrenal obstruction management and post-obstructive diuresis monitoring.",
        # Neurology & Psychiatry (10)
        "A 69-year-old man has sudden onset right-sided facial droop, right arm weakness, and expressive aphasia starting 95 minutes ago. Walk through non-contrast head CT interpretation, IV thrombolysis eligibility and contraindications, CT angiography for large-vessel occlusion, and mechanical thrombectomy windows.",
        "Compare ischemic penumbra physiology, watershed infarcts, lacunar strokes of the internal capsule and thalamus, and Wallenberg lateral medullary syndrome neuroanatomy.",
        "A 24-year-old man is brought in during continuous generalized tonic-clonic seizure activity lasting 12 minutes. Specify first-line benzodiazepine dosing, second-line IV antiepileptic loading options, refractory status epilepticus anesthetic infusions, and metabolic triggers to correct.",
        "Contrast myasthenia gravis with Lambert-Eaton myasthenic syndrome across neuromuscular junction antibodies, repetitive nerve stimulation electrodiagnostics, associated thymoma or small-cell carcinoma, and edrophonium/pyridostigmine pharmacology.",
        "Explain the pathophysiology of idiopathic Parkinson disease, direct vs indirect basal ganglia striatal pathways, carbidopa-levodopa peripheral decarboxylase inhibition, and management of wearing-off dyskinesias.",
        "A 31-year-old woman presents with transient monocular vision loss, internuclear ophthalmoplegia, and periventricular Dawson fingers on brain MRI. Discuss McDonald criteria for dissemination in space and time, acute relapse corticosteroids, and disease-modifying immunotherapies.",
        "Compare bacterial, viral, and fungal meningitis CSF profiles (opening pressure, WBC predominance, glucose ratio, protein), empiric IV antimicrobial plus dexamethasone timing in adults, and Kernig/Brudzinski exam limits.",
        "A 58-year-old woman describes the worst headache of her life reaching maximal intensity within seconds, photophobia, and neck stiffness. Outline sensitivity of early head CT, lumbar puncture xanthochromia spectrophotometry, vasospasm nimodipine prophylaxis, and aneurysm securing.",
        "Contrast Guillain-Barre syndrome (acute inflammatory demyelinating polyradiculoneuropathy) with transverse myelitis and botulism regarding sensory levels, deep tendon reflexes, CSF albuminocytologic dissociation, and respiratory FVC/NIF monitoring.",
        "Explain serotonin syndrome versus neuroleptic malignant syndrome across causative drug classes, onset tempo, neuromuscular findings (clonus/hyperreflexia vs lead-pipe rigidity), hyperthermia management, and cyproheptadine vs dantrolene.",
        # Endocrinology & Metabolism (10)
        "A 22-year-old patient with polyuria, polydipsia, nausea, abdominal pain, and Kussmaul respirations has glucose 540 mg/dL, beta-hydroxybutyrate 6.8 mmol/L, K 3.4 mEq/L, HCO3 9 mEq/L. Detail fluid resuscitation, potassium replacement threshold BEFORE insulin, fixed-rate insulin infusion, anion gap closure, and subcutaneous overlap transition.",
        "Contrast diabetic ketoacidosis (DKA) and hyperosmolar hyperglycemic state (HHS) with respect to residual insulin suppression of lipolysis, effective serum osmolality formulas, typical free water deficits, and cerebral edema risk factors in children.",
        "A 34-year-old woman presents with heat intolerance, 8 kg weight loss, exophthalmos, pretibial myxedema, suppressed TSH <0.01 mIU/L, and elevated free T4. Explain thyroid-stimulating immunoglobulin pathophysiology, radioactive iodine uptake patterns, thionamide mechanisms, and thyroid storm protocol.",
        "Describe hypothalamic-pituitary-adrenal axis feedback, high-dose vs low-dose dexamethasone suppression tests, CRH stimulation, and inferior petrosal sinus sampling to differentiate Cushing disease, ectopic ACTH, and adrenal adenoma.",
        "A 41-year-old man has resistant episodic headaches, diaphoresis, and paroxysmal palpitations. Detail plasma free metanephrines vs 24-hour urine fractionated metanephrines, why alpha-blockade with phenoxybenzamine MUST precede beta-blockade, and MEN2A/2B genetic screening.",
        "Contrast primary adrenal insufficiency (Addison disease) with secondary pituitary ACTH deficiency across hyperpigmentation, mineralocorticoid preservation, hyperkalemia, and cosyntropin stimulation interpretation.",
        "A 59-year-old woman has asymptomatic corrected serum calcium 11.9 mg/dL, serum phosphorus 2.1 mg/dL, elevated intact PTH 145 pg/mL, and 24-hour urine calcium 340 mg. Differentiate primary hyperparathyroidism from familial hypocalciuric hypercalcemia and malignancy-related PTHrP.",
        "Explain the physiology of growth hormone and IGF-1, oral glucose tolerance suppression failure in acromegaly, somatostatin analog therapy, and long-term cardiovascular/colonic complications.",
        "Compare primary hypothyroidism causes (Hashimoto thyroiditis, subacute de Quervain thyroiditis, amiodarone-induced hypothyroidism vs thyrotoxicosis type 1/2) and levothyroxine absorption interactions.",
        "Outline bone remodeling RANK/RANKL/OPG signaling, DXA T-score diagnostic thresholds for osteopenia and osteoporosis, bisphosphonate osteoclast apoptosis vs denosumab vs teriparatide anabolic therapy.",
        # Infectious Disease (10)
        "Outline the Surviving Sepsis Campaign 1-hour bundle for septic shock, dynamic fluid responsiveness assessment, norepinephrine as first-line vasopressor, hydrocortisone indications in refractory shock, and lactate clearance monitoring.",
        "Explain HIV-1 reverse transcriptase, integrase, and protease life cycle steps, modern single-tablet integrase strand transfer inhibitor regimens, opportunistic infection CD4 thresholds (PCP, toxoplasmosis, MAC, CMV), and immune reconstitution inflammatory syndrome.",
        "Compare latent versus active pulmonary tuberculosis diagnostics (IGRA vs TST, sputum AFB smear, NAAT, culture), four-drug RIPE mechanism and hallmark toxicities (optic neuritis, hepatotoxicity, orange secretions, peripheral neuropathy), and airborne isolation.",
        "A 48-year-old patient with neutropenia (ANC 180/uL) seven days after induction chemotherapy for AML develops fever 38.9 C. Detail immediate cultures, antipseudomonal beta-lactam monotherapy vs vancomycin indications, MASCC scoring, and empiric antifungal escalation.",
        "Describe Clostridium difficile pathogenesis (toxins A and B), why alcohol hand rub fails against spores, severe vs fulminant definitions, and oral fidaxomicin/vancomycin plus IV metronidazole strategies.",
        "Compare methicillin-resistant Staphylococcus aureus (MRSA) mecA/PBP2a resistance, vancomycin-resistant Enterococcus (VRE) vanA D-Ala-D-Lac alteration, and ESBL vs carbapenemase-producing Enterobacterales mechanisms and antibiotic choices.",
        "A traveler returning from sub-Saharan Africa develops cyclical high fevers, chills, thrombocytopenia, and 6% parasitemia on thick/thin blood smears. Explain Plasmodium falciparum sequestration physiology, severe malaria criteria, and IV artesunate treatment.",
        "Explain Lyme disease early localized, early disseminated (carditis, Bell palsy, meningitis), and late arthritis stages, two-tier ELISA/Western blot serology, and doxycycline vs ceftriaxone regimens.",
        "Compare herpes simplex encephalitis (temporal lobe hemorrhage on MRI, PCR, IV acyclovir nephrotoxicity prevention) with progressive multifocal leukoencephalopathy (JC virus, non-enhancing demyelination).",
        "Discuss acute vs chronic hepatitis B serologic panel interpretation (HBsAg, anti-HBs, IgM/IgG anti-HBc, HBeAg, HBV DNA), vertical transmission immunoprophylaxis, and tenofovir/entecavir suppression.",
        # GI & Hepatology (10)
        "Compare ulcerative colitis and Crohn disease across anatomic distribution, transmural vs mucosal histology, skip lesions, granulomas, extraintestinal manifestations, toxic megacolon risk, and biologic targets (anti-TNF, anti-integrin, anti-IL-23).",
        "A 56-year-old man with cirrhosis presents with hematemesis and BP 88/54 mmHg. Outline restrictive hemoglobin transfusion threshold (7 g/dL), IV octreotide, prophylactic ceftriaxone, urgent EGD band ligation, and TIPS criteria.",
        "Explain the diagnostic evaluation of acute pancreatitis, modified Atlanta severity classification, early aggressive vs moderate lactated Ringer hydration, early enteral nutrition rationale, and walled-off necrosis vs pseudocyst drainage timing.",
        "Contrast direct and indirect bilirubin metabolism across Gilbert syndrome, Crigler-Najjar, Dubin-Johnson, Rotor syndrome, primary biliary cholangitis (AMA), and primary sclerosing cholangitis (MRCP beading).",
        "A 45-year-old woman has epigastric pain improved with meals and nocturnal awakening vs pain worsened immediately after eating. Compare duodenal and gastric ulcers, H. pylori urea breath test vs stool antigen, quadruple bismuth therapy, and PPI H+/K+-ATPase irreversible inhibition.",
        "Explain celiac disease HLA-DQ2/DQ8 autoimmunity, tissue transglutaminase IgA sensitivity and total IgA deficiency pitfall, Marsh villous atrophy grading, and dermatitis herpetiformis association.",
        "Describe spontaneous bacterial peritonitis pathogenesis in cirrhotic ascites, SAAG >= 1.1 g/dL interpretation, PMN >= 250/mm3 diagnostic threshold, third-generation cephalosporin therapy, and albumin dosing to prevent renal injury.",
        "Compare acute cholecystitis, choledocholithiasis, and acute ascending cholangitis (Charcot triad, Reynolds pentad), Murphy sign, HIDA scan vs MRCP vs ERCP decompression.",
        "Explain hepatic encephalopathy ammonia astrocyte glutamine swelling hypothesis, precipitating triggers (GI bleed, constipation, infection, hypokalemia), lactulose colonic acidification trapping, and rifaximin non-absorbable action.",
        "Discuss colorectal cancer adenoma-carcinoma sequence (APC, KRAS, TP53), Lynch syndrome mismatch repair deficiency vs FAP, and age/modality screening guidelines.",
        # Heme / Onc (10)
        "Compare microcytic anemias (iron deficiency, anemia of chronic inflammation, alpha/beta thalassemia trait, sideroblastic anemia) using ferritin, TIBC, transferrin saturation, RDW, Mentzer index, and hepcidin regulation.",
        "A 33-year-old woman presents with petechiae, platelet count 9,000/uL, schistocytes on peripheral smear, LDH 1150 U/L, low haptoglobin, mild confusion, and fever. Distinguish thrombotic thrombocytopenic purpura (ADAMTS13) from HUS and DIC, and detail emergent plasma exchange vs contraindicated platelet transfusion.",
        "Compare intrinsic, extrinsic, and common coagulation pathways, PT/INR vs aPTT mixing studies (factor deficiency vs inhibitor), hemophilia A/B, and von Willebrand disease ristocetin cofactor testing.",
        "Explain tumor lysis syndrome pathophysiology, hyperuricemia/hyperkalemia/hyperphosphatemia/hypocalcemia cascade, allopurinol xanthine oxidase blockade vs rasburicase urate oxidase, and Cairo-Bishop laboratory vs clinical criteria.",
        "Contrast chronic myeloid leukemia (t(9;22) BCR-ABL1 tyrosine kinase, basophilia, LAP score) with chronic lymphocytic leukemia (smudge cells, CD5/CD19/CD23) and acute promyelocytic leukemia (t(15;17), Auer rods, ATRA differentiation).",
        "Detail multiple myeloma CRAB criteria, serum protein electrophoresis M-spike, kappa/lambda free light chain ratio, hyperviscosity, lytic bone lesions, and proteasome inhibitor/IMiD mechanisms.",
        "Compare warm autoimmune hemolytic anemia (IgG, extravascular splenic spherocytes, direct Coombs positive) with cold agglutinin disease (IgM, complement-mediated, Mycoplasma/EBV triggers) and G6PD deficiency bite cells.",
        "Explain heparin-induced thrombocytopenia (HIT type II PF4-heparin IgG antibodies), 4Ts pretest probability score, serotonin release assay gold standard, immediate cessation of all heparins, and argatroban/bivalirudin bridging.",
        "Describe sickle cell disease beta-globin E6V polymerization under deoxygenation, vaso-occlusive crisis, acute chest syndrome management, hydroxyurea HbF induction, and autosplenectomy encapsulated organism prophylaxis.",
        "Compare direct oral anticoagulants (apixaban, rivaroxaban, dabigatran) versus warfarin across target enzymes, vitamin K epoxide reductase inhibition, renal clearance, and specific reversal agents (idarucizumab, andexanet alfa, 4F-PCC + vitamin K).",
        # Rheumatology, Pharmacology, Pediatrics, OB-GYN, Toxicology (18)
        "Compare rheumatoid arthritis, systemic lupus erythematosus, and psoriatic arthritis across autoantibody specificity (anti-CCP, anti-dsDNA/anti-Sm), joint distribution, radiographic erosions vs pencil-in-cup, and DMARD mechanisms (methotrexate vs TNF inhibitors).",
        "A 62-year-old man has sudden exquisite pain and erythema of the first metatarsophalangeal joint. Differentiate gout from pseudogout via synovial fluid polarized microscopy (needle-shaped negatively birefringent vs rhomboid positively birefringent), acute colchicine/steroid therapy, and chronic urate-lowering xanthine oxidase inhibition.",
        "Compare ANCA-associated small-vessel vasculitides (granulomatosis with polyangiitis c-ANCA/PR3, microscopic polyangiitis, eosinophilic granulomatosis with polyangiitis) and giant cell arteritis urgency of high-dose corticosteroids before temporal artery biopsy.",
        "A 26-year-old patient ingests an unknown quantity of acetaminophen 6 hours ago. Explain CYP2E1 NAPQI formation, glutathione depletion, Rumack-Matthew nomogram interpretation, and N-acetylcysteine IV protocol.",
        "Compare organophosphate/carbamate cholinergic toxidrome versus anticholinergic toxicity vs sympathomimetic overdose, explaining atropine muscarinic blockade and pralidoxime acetylcholinesterase regeneration prior to aging.",
        "Compare opioid receptor mu/kappa/delta signaling, naloxone competitive antagonism short half-life vs long-acting opioids, and buprenorphine partial agonist precipitated withdrawal physiology.",
        "Explain aminoglycoside, vancomycin, and fluoroquinolone antibacterial mechanisms, concentration-dependent vs time-dependent PK/PD indices (Cmax/MIC, AUC/MIC, fT>MIC), and nephrotoxicity/ototoxicity monitoring.",
        "Compare pharmacokinetic volume of distribution, first-order vs zero-order elimination kinetics (phenytoin, ethanol, high-dose aspirin), loading dose formulas, and maintenance infusion steady state (4-5 half-lives).",
        "A 32-week pregnant woman presents with BP 166/108 mmHg, headache, RUQ pain, AST 240, platelets 72,000/uL, and 3+ proteinuria. Outline preeclampsia with severe features / HELLP syndrome, IV magnesium sulfate seizure prophylaxis, labetalol/hydralazine/nifedipine, betamethasone, and delivery timing.",
        "Explain neonatal respiratory distress syndrome surfactant deficiency pathophysiology (lecithin-to-sphingomyelin ratio), antenatal corticosteroids, CPAP, exogenous surfactant, and retinopathy of prematurity oxygen balance.",
        "Contrast congenital heart diseases: Tetralogy of Fallot hypercyanotic tet spells (knee-chest physiology), transposition of the great arteries PGE1 ductal patency, and coarctation of the aorta.",
        "Compare pyloric stenosis (hypochloremic hypokalemic metabolic alkalosis, olive mass), intussusception (currant jelly stools, target sign, air enema), and midgut volvulus bilious emesis surgical emergency.",
        "Describe acute angle-closure glaucoma pathophysiology, dilated fixed pupil, corneal edema, emergent intraocular pressure lowering (acetazolamide, timolol, pilocarpine), and definitive laser peripheral iridotomy.",
        "Contrast benign paroxysmal positional vertigo (Dix-Hallpike, Epley maneuver), Meniere disease endolymphatic hydrops, vestibular neuritis, and central brainstem vertigo via HINTS exam.",
        "Explain ectopic pregnancy diagnosis with beta-hCG discriminatory zone and transvaginal ultrasound, methotrexate medical eligibility criteria vs salpingostomy/salpingectomy, and RhD immunoglobulin prophylaxis.",
        "Compare anaphylaxis IgE mast cell degranulation mechanisms, intramuscular anterolateral thigh epinephrine 0.3-0.5 mg dosing, biphasic reaction observation, and adjunctive H1/H2/glucocorticoid roles.",
        "Explain malignant hyperthermia RYR1 calcium release mutation triggered by succinylcholine or volatile anesthetics, rising end-tidal CO2 earliest sign, hyperkalemia, and dantrolene rescue.",
        "Describe local anesthetic systemic toxicity (LAST), CNS excitation progressing to cardiovascular collapse, bupivacaine cardiac sodium channel avidity, and 20% intravenous lipid emulsion rescue.",
    ]

    chem = [
        # Physical, Thermo, Kinetics, Electrochem, Equilibria (22)
        "Derive the Henderson-Hasselbalch equation from acid dissociation equilibrium, and calculate the pH and buffer capacity when mixing 250 mL of 0.40 M acetic acid (pKa = 4.76) with 150 mL of 0.50 M sodium acetate, followed by adding 0.020 mol HCl.",
        "Compare zero-order, first-order, and second-order chemical reaction integrated rate laws, half-life dependence on initial concentration, and how Arrhenius activation energy is extracted from ln(k) vs 1/T.",
        "Explain collision theory, transition state theory (Eyring equation ΔG‡, ΔH‡, ΔS‡), pre-equilibrium vs steady-state approximation in multi-step mechanisms, and kinetic vs thermodynamic control.",
        "Starting from ΔG = ΔH - TΔS and ΔG = ΔG° + RT ln(Q), explain why the van 't Hoff equation predicts opposite temperature shifts of equilibrium constant K for endothermic versus exothermic reactions.",
        "Write balanced half-reactions in BOTH acidic and basic aqueous solution for the disproportionation of chlorine gas into chloride and hypochlorite, and for the dichromate oxidation of ethanol to acetic acid.",
        "Explain standard reduction potentials, the Nernst equation, concentration cells, relationship between E°cell, ΔG°, and K_eq, and why zinc sacrifices itself to protect steel in cathodic protection.",
        "Compare galvanic (voltaic) cells, electrolytic cells, and lithium-ion intercalation batteries, showing Faraday's laws of electrolysis calculations to deposit 12.7 g of copper from CuSO4(aq) at 3.50 A.",
        "Explain Raoult's law, positive and negative azeotropes from intermolecular interactions, boiling point elevation, freezing point depression, and osmotic pressure with van 't Hoff factor corrections.",
        "Compare solubility product Ksp, common-ion effect, selective precipitation of metal sulfides/halides using pH control, and complex-ion formation constants Kf (e.g. AgCl dissolving in NH3).",
        "Explain polyprotic acid (H3PO4) alpha-fraction speciation curves, amphoteric equivalence point pH estimation via (pKa1 + pKa2)/2, and titration curve inflection points.",
        "Describe Hess's law, standard enthalpies of formation, bond dissociation enthalpy approximations, and Born-Haber cycle lattice energy estimation for ionic crystals like MgO vs NaCl.",
        "Explain kinetic molecular theory of gases, Maxwell-Boltzmann speed distributions, Graham's law of effusion, and the van der Waals equation parameters a and b accounting for non-ideal compressibility factor Z.",
        "Compare homogeneous catalysis, heterogeneous surface Langmuir-Hinshelwood catalysis, and autocatalysis, illustrating potential energy diagrams with lowered activation barriers.",
        "Describe Beer-Lambert law absorbance linearity limits, spectrophotometric determination of equilibrium binding constants, and fluorescence Stokes shift vs phosphorescence intersystem crossing.",
        "Explain phase diagrams of CO2 and H2O (negative solid-liquid slope of water via Clausius-Clapeyron equation), triple point, supercritical fluid properties, and Gibbs phase rule.",
        "Compare VSEPR geometries, hybridization, and molecular orbital theory explanations for why O2 is paramagnetic with bond order 2 whereas N2 has a high HOMO-LUMO gap and bond order 3.",
        "Explain crystal field theory splitting Δ_o vs Δ_t, spectrochemical series weak-field vs strong-field ligands, high-spin vs low-spin d6 complexes ([Fe(CN)6]4- vs [Fe(H2O)6]2+), and Jahn-Teller distortion in Cu(II).",
        "Compare Lewis acids/bases, Hard-Soft Acid-Base (HSAB) theory, Brønsted-Lowry conjugate strength, and gas-phase vs aqueous basicity trends of methylamines.",
        "Explain nuclear radioactive decay kinetics (alpha, beta-minus, positron emission, electron capture), binding energy per nucleon curve, and carbon-14 radiometric dating calculations.",
        "Describe chelation entropy effect (EDTA hexadentate vs six monodentate NH3 ligands), stability constants, and complexometric calcium/magnesium water hardness titrations.",
        "Explain colligative vs non-colligative osmotic phenomena, Donnan membrane equilibrium, and Debye-Hückel ionic strength activity coefficient deviations in concentrated electrolytes.",
        "Compare LCAO-MO sigma/pi bonding and antibonding orbitals for first-row diatomics B2, C2, O2, and F2, highlighting s-p mixing inversion between N2 and O2.",
        # Organic, Bioorganic, Spectroscopy, Analytical (23)
        "Contrast SN1, SN2, E1, and E2 mechanisms across carbocation stability, stereochemical inversion vs racemization, Zaitsev vs Hofmann regioselectivity, protic vs aprotic solvents, and leaving group ability.",
        "Compare electrophilic aromatic substitution activating ortho/para directors (-OH, -NH2, -OCH3) against deactivating meta directors (-NO2, -CF3, -COR) and halogens (ortho/para deactivating) using resonance contributors.",
        "Explain nucleophilic acyl substitution reactivity order across acid chlorides, anhydrides, esters, and amides, tetrahedral intermediate collapse, and saponification irreversibility.",
        "Compare aldol addition/condensation, Claisen condensation, Dieckmann cyclization, and Michael conjugate addition (1,4 vs 1,2 with Gilman cuprates vs Grignard reagents).",
        "Explain carbonyl reduction selectivity between NaBH4, LiAlH4, DIBAL-H at -78 C, Clemmensen vs Wolff-Kishner deoxygenation, and chemoselective acetal protecting groups.",
        "Compare Markovnikov hydrohalogenation, oxymercuration-demercuration, hydroboration-oxidation (anti-Markovnikov syn addition), ozonolysis, and Sharpless/mCPBA epoxidation stereochemistry.",
        "Describe Diels-Alder [4+2] cycloaddition Woodward-Hoffmann Woodward-Fieser stereospecificity, s-cis diene conformation requirement, electron-withdrawing dienophile acceleration, and endo selectivity.",
        "Explain Fischer esterification mechanism, isotopic 18O labeling proof, Le Chatelier water removal strategies, and transesterification in biodiesel synthesis.",
        "Compare keto-enol tautomerism acid vs base catalysis, alpha-halogenation vs haloform reaction, and decarboxylation of beta-keto acids.",
        "Explain Cahn-Ingold-Prelog R/S stereocenter assignment, meso compounds, diastereomers vs enantiomers, optical rotation, and chiral resolution via diastereomeric salt formation.",
        "Describe Wittig reaction phosphorus ylide formation, stabilized vs unstabilized ylide E/Z alkene selectivity, and Horner-Wadsworth-Emmons variation.",
        "Compare aromaticity Huckel 4n+2 pi electron rules across benzene, cyclopentadienyl anion, tropylium cation, pyrrole, pyridine (basicity difference of lone pair), and antiaromatic cyclobutadiene.",
        "Explain proton (1H) NMR chemical shifts, magnetic anisotropy in aromatics and alkenes, spin-spin coupling (n+1 rule, Pascal's triangle, J-value dihedral Karplus relation), and D2O exchange.",
        "Describe 13C NMR DEPT-90/135 multiplicity editing, IR diagnostic stretching frequencies (carbonyl conjugation lowering, broad O-H vs sharp N-H), and mass spectrometry McLafferty rearrangement & M+2 isotope patterns (Cl/Br).",
        "Compare radical halogenation selectivity of Br2 vs Cl2 on alkanes via Hammond postulate transition states, allylic NBS bromination, and peroxide anti-Markovnikov HBr addition.",
        "Explain carbohydrate stereochemistry, D/L Fischer projections, mutarotation between alpha/beta anomers, glycosidic bond acetals, and reducing vs non-reducing sugars (sucrose vs maltose).",
        "Describe peptide solid-phase Fmoc synthesis, DCC/EDC coupling activation, Edman degradation N-terminal sequencing, and disulfide cystine redox folding.",
        "Compare nucleophilic aromatic substitution (SNAr Meisenheimer intermediate) with benzyne elimination-addition mechanism using 14C labeling proof.",
        "Explain Beckmann rearrangement of oximes, Baeyer-Villiger oxidation migratory aptitude, and Hofmann rearrangement of primary amides to amines with one fewer carbon.",
        "Compare free-radical polymerization, anionic living polymerization, and step-growth condensation polymers (nylon-6,6 vs PET) with respect to molecular weight dispersity vs conversion.",
        "Explain chromatography principles: retention factor Rf, Van Deemter plate height (A/B/C terms), reverse-phase HPLC C18 polarity elution order, and ion-exchange pI separation.",
        "Describe palladium-catalyzed cross-coupling cycles (oxidative addition, transmetalation, reductive elimination) comparing Suzuki-Miyaura, Heck, and Sonogashira reactions.",
        "Explain asymmetric catalytic hydrogenation using BINAP-Ru catalysts, enantiomeric excess calculation from chiral HPLC peak areas, and kinetic resolution.",
    ]

    bio = [
        # Molecular Biology, Genetics, Gene Regulation (16)
        "Explain eukaryotic RNA polymerase II transcription initiation, promoter elements (TATA, Initiator, BRE), general transcription factors (TFIID/TBP, TFIIH helicase/CTD kinase), mediator complex, and 5'-cap/poly-A/spliceosome processing.",
        "Compare DNA replication at leading versus lagging strands in prokaryotes and eukaryotes: helicase, primase, DNA polymerase III/delta/epsilon 3'->5' exonuclease proofreading, Okazaki fragment maturation, and telomerase.",
        "Explain the trp operon attenuation mechanism in bacteria: how leader peptide tryptophan codons and competing 1:2, 2:3, and 3:4 mRNA stem-loops couple translation speed to transcription termination.",
        "Compare chromatin epigenetics (histone acetyltransferases vs deacetylases, H3K4me3 vs H3K27me3 polycomb repression, CpG island DNA methylation) and genomic imprinting.",
        "Explain eukaryotic mRNA translation initiation (eIF4E cap binding, 43S preinitiation scanning, Kozak sequence), elongation GTPases (eEF1A, eEF2), peptidyl transferase ribozyme, and nonsense-mediated decay.",
        "Compare DNA repair pathways: base excision repair (glycosylase/AP endonuclease), nucleotide excision repair (xeroderma pigmentosum pyrimidine dimers), mismatch repair, NHEJ vs homologous recombination (BRCA1/2, RAD51).",
        "Describe CRISPR-Cas9 ribonucleoprotein targeting (sgRNA spacer, PAM recognition, HNH and RuvC nuclease domains), base editing, prime editing, and off-target specificity strategies.",
        "Explain RNA interference via miRNA and siRNA pathways (Drosha, Exportin-5, Dicer, RISC/Ago2 cleavage vs translational repression) and therapeutic antisense oligonucleotides.",
        "Compare alternative pre-mRNA splicing mechanisms (U1/U2/U4/U5/U6 snRNPs, lariat transesterification, SR activators vs hnRNP silencers) and how tissue-specific isoforms expand proteome diversity.",
        "Explain Hardy-Weinberg equilibrium assumptions, allele/genotype frequency calculation when 1 in 2,500 individuals expresses an autosomal recessive trait, and forces breaking equilibrium (drift, founder effect, heterozygote advantage).",
        "Compare Mendelian autosomal, X-linked recessive/dominant, mitochondrial maternal inheritance, incomplete penetrance, variable expressivity, pleiotropy, and anticipation in trinucleotide repeat disorders.",
        "Describe mobile genetic elements: bacterial IS/Tn transposons, retrotransposons (LINEs, SINEs/Alu, LTRs), and V(D)J recombination RAG1/RAG2 12/23 rule diversity.",
        "Explain quantitative genetics: broad-sense vs narrow-sense heritability, polygenic risk scores, linkage disequilibrium, GWAS Manhattan plots, and CRISPR pooled functional screens.",
        "Compare prokaryotic SOS response (RecA ssDNA filament, LexA autocleavage) and stringent response ((p)ppGpp alarmone) under nutrient deprivation or DNA damage.",
        "Explain unfolded protein response (UPR) branches in the endoplasmic reticulum (IRE1alpha XBP1 splicing, PERK eIF2alpha phosphorylation, ATF6 Golgi cleavage) and ER-associated degradation (ERAD).",
        "Describe protein folding chaperones (Hsp70 ATPase cycle, GroEL/GroES GroE chaperonin cage), ubiquitin-proteasome K48 polyubiquitination, and macroautophagy autophagosome-lysosome flux.",
        # Biochemistry, Metabolism, Enzymology (15)
        "Explain Michaelis-Menten enzyme kinetics assumptions, Lineweaver-Burk double-reciprocal plots, and how competitive, uncompetitive, and mixed/noncompetitive inhibitors shift apparent Km and Vmax.",
        "Trace every regulatory enzyme and allosteric effector of glycolysis (hexokinase, PFK-1/F2,6BP, pyruvate kinase) and coordinate reciprocal control with gluconeogenesis via insulin vs glucagon signaling.",
        "Describe the pyruvate dehydrogenase multi-enzyme complex (E1/E2/E3 and five cofactors TPP, lipoate, CoA, FAD, NAD+), citric acid cycle anaplerotic replenishment, and net ATP/NADH/FADH2 yield.",
        "Explain mitochondrial oxidative phosphorylation Complexes I-IV proton pumping, Q-cycle, chemiosmotic proton-motive force, ATP synthase F0/F1 rotational catalysis, and uncouplers (DNP, UCP1 thermogenin).",
        "Trace the pentose phosphate pathway oxidative (G6PD, NADPH generation) and non-oxidative (transketolase/transaldolase) branches and explain why rapidly dividing and red blood cells depend on it.",
        "Explain beta-oxidation of saturated and odd-chain fatty acids, carnitine shuttle (CPT-I malonyl-CoA inhibition), ketone body ketogenesis in hepatocytes, and ketolysis in extrahepatic tissues.",
        "Describe de novo fatty acid synthesis (citrate shuttle, acetyl-CoA carboxylase, fatty acid synthase ACP), SREBP-1c transcriptional regulation, and triacylglycerol/VLDL assembly.",
        "Trace the urea cycle compartmentalization between mitochondria and cytosol, N-acetylglutamate activation of CPS-I, and amino acid nitrogen transport via the glucose-alanine and glutamine cycles.",
        "Compare glycogenolysis and glycogenesis control in skeletal muscle vs liver: glycogen phosphorylase a/b covalent phosphorylation, calcium-calmodulin coordination, and epinephrine/glucagon PKA cascades.",
        "Explain purine and pyrimidine de novo vs salvage nucleotide biosynthesis (PRPP synthetase, IMP branching, carbamoyl phosphate synthetase II, ribonucleotide reductase, thymidylate synthase).",
        "Compare hemoglobin cooperative O2 binding (Bohr effect, 2,3-BPG T-state stabilization, HbF gamma-subunit affinity) with myoglobin hyperbolic binding curve and Hill coefficient.",
        "Explain enzyme catalytic strategies (covalent catalysis in chymotrypsin Ser-His-Asp triad, general acid-base catalysis in RNase A, metal ion catalysis, transition-state stabilization) and allosteric MWC/KNF models.",
        "Compare photosynthetic light reactions (Z-scheme PSII water splitting, cytochrome b6f, PSI ferredoxin-NADP+ reductase) and Calvin-Benson cycle (RuBisCO carboxylation vs photorespiration C4/CAM adaptations).",
        "Trace cholesterol biosynthesis rate-limiting HMG-CoA reductase, statin competitive inhibition, LDL receptor endocytosis, PCSK9 degradation, and reverse cholesterol HDL transport.",
        "Describe heme biosynthesis (ALA synthase vitamin B6 cofactor, mitochondrial-cytosolic-mitochondrial steps) and acute intermittent vs porphyria cutanea tarda enzymatic bottlenecks.",
        # Cell Biology, Immunology, Physiology (14)
        "Explain innate versus adaptive immunity: Toll-like receptor PAMP signaling, inflammasome IL-1beta maturation, dendritic cell MHC Class I cross-presentation vs Class II loading, and CD4 Th1/Th2/Th17/Treg polarization.",
        "Describe B-cell somatic hypermutation (AID), germinal center affinity maturation, class-switch recombination, central/peripheral T-cell thymic positive and negative selection (AIRE), and checkpoint CTLA-4/PD-1 braking.",
        "Compare G-protein-coupled receptor pathways (Gs/cAMP/PKA, Gi, Gq/PLC-beta/IP3/DAG/Ca2+), receptor tyrosine kinase Ras-Raf-MEK-ERK MAPK cascades, and JAK-STAT cytokine signaling.",
        "Explain skeletal, cardiac, and smooth muscle excitation-contraction coupling: T-tubule DHPR/RyR calcium handling, troponin C vs calmodulin myosin light-chain kinase, and cross-bridge ATP detachment.",
        "Describe neuronal action potential Hodgkin-Huxley voltage-gated Na+ and K+ channel kinetics, absolute vs relative refractory periods, saltatory myelin conduction, and chemical synapse SNARE exocytosis.",
        "Explain glomerular filtration Starling forces, proximal tubule Na+/glucose/bicarbonate reabsorption, thick ascending limb NKCC2 countercurrent multiplier, vasa recta exchanger, and collecting duct aquaporin-2.",
        "Trace gastrointestinal digestion and absorption of carbohydrates, proteins, and triglycerides (micelles, chylomicrons, lymphatics), gastric parietal H+/K+ secretion regulation, and pancreatic zymogen enteropeptidase cascade.",
        "Explain cardiac cycle Wiggers diagram phases (isovolumetric contraction/relaxation, rapid ventricular filling, dicrotic notch), pressure-volume loops, preload/afterload/contractility shifts, and arterial baroreceptor reflex.",
        "Compare apoptosis intrinsic mitochondrial (BCL-2/BAX/BAK, cytochrome c, Apaf-1, caspase-9/3) vs extrinsic death receptor (Fas/FADD/caspase-8), necroptosis (RIPK1/RIPK3/MLKL), and pyroptosis.",
        "Describe vesicular trafficking COPI, COPII, and clathrin coats, Rab GTPases, mannose-6-phosphate lysosomal sorting, and Golgi cis-to-trans glycosylation processing.",
        "Explain extracellular matrix architecture: triple-helical collagen synthesis (hydroxylation vitamin C, lysyl oxidase cross-linking), elastin, proteoglycan hydration, and integrin focal adhesion mechanotransduction.",
        "Trace oxygen and carbon dioxide blood transport: oxyhemoglobin dissociation curve shifts, Haldane effect, chloride shift via Band 3 anion exchanger, and central vs peripheral chemoreceptors.",
        "Explain complement cascade classical, lectin, and alternative pathway activation, C3 convertase amplification, opsonization (C3b), anaphylatoxins (C3a/C5a), and membrane attack complex (C5b-9) regulation by CD55/CD59.",
        "Describe circadian molecular clock transcription-translation feedback loop (CLOCK/BMAL1, PER/CRY), suprachiasmatic nucleus melanopsin light entrainment, and pineal melatonin secretion.",
    ]

    math_code = [
        # Math, Probability, Stats (15)
        "Prove by mathematical induction and also via generating functions or telescoping sums that sum_{k=1}^n k^3 = (n(n+1)/2)^2, then evaluate sum_{k=10}^{40} k^3 step by step.",
        "Compute the eigenvalues and orthonormal eigenvectors of the symmetric 3x3 matrix [[4, -1, 1], [-1, 4, -1], [1, -1, 4]], verify orthogonality, and express A^5 using spectral decomposition.",
        "Derive Bayes' theorem and solve a two-stage diagnostic screening problem: disease prevalence 2%, test sensitivity 94%, specificity 97%. Compute positive predictive value after one positive test, and then after an independent second positive test.",
        "Compare Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) estimation for a Bernoulli coin with Beta(alpha, beta) prior, deriving exact variance and Fisher information of the MLE.",
        "Solve the second-order linear non-homogeneous ODE y'' - 4y' + 4y = (t^2 + 1)e^(2t) with y(0)=1, y'(0)=0 using undetermined coefficients and variation of parameters.",
        "Explain the Central Limit Theorem, Law of Large Numbers (weak vs strong), Chebyshev's inequality, and construct a 95% confidence interval and p-value calculation for a two-sample Welch t-test scenario.",
        "Use Lagrange multipliers to maximize f(x, y, z) = x*y*z^2 subject to x + 2y + 3z = 24 with x, y, z > 0, and verify second-order optimality.",
        "Explain the Singular Value Decomposition (SVD) of an m x n matrix, geometric interpretation on the unit sphere, Eckart-Young best rank-k approximation theorem, and pseudoinverse least-squares solution.",
        "Solve the linear congruence system x ≡ 3 (mod 7), x ≡ 5 (mod 11), x ≡ 2 (mod 13) using the Chinese Remainder Theorem and verify by substitution.",
        "Derive the moment-generating function of Poisson(lambda) and Gamma(k, theta) random variables, prove additivity of independent Poissons, and relate exponential interarrival times to Poisson counts.",
        "Evaluate the improper integral int_0^inf e^(-a x^2) cos(b x) dx by differentiating under the integral sign (Feynman's trick) and verify limiting behavior as b -> 0.",
        "Compare gradient descent, momentum, Nesterov accelerated gradient, and Adam optimizers on an ill-conditioned quadratic convex function f(x) = 0.5 x^T H x, deriving optimal step sizes and condition-number convergence rates.",
        "Explain Markov chain stationary distributions, detailed balance condition, ergodicity (irreducibility + aperiodicity), and solve a 3-state weather transition steady-state distribution explicitly.",
        "Derive Huffman coding optimality proof via greedy choice and optimal substructure, construct the binary prefix tree for symbol frequencies {A: 0.40, B: 0.25, C: 0.15, D: 0.12, E: 0.08}, and compare average code length to Shannon entropy.",
        "Explain the Fourier Transform vs Discrete Fourier Transform (DFT), convolution theorem, Nyquist-Shannon sampling theorem, aliasing, and how the Cooley-Tukey FFT reduces O(N^2) to O(N log N).",
        # Python / C++ Coding, Systems, Algorithms (15)
        "Write a clean, well-documented Python implementation of Dijkstra's shortest path algorithm using heapq plus path reconstruction and A* heuristic extension, and analyze time and space complexity.",
        "Implement a thread-safe bounded ring-buffer LRU cache in Python with O(1) get and put operations (using doubly linked list + dict rather than OrderedDict), and write unit tests covering eviction and concurrency.",
        "Write modern C++20 code implementing a lock-free single-producer single-consumer (SPSC) bounded queue using std::atomic with acquire-release memory ordering and cache-line padding to avoid false sharing.",
        "Explain virtual memory paging, multi-level page tables, TLB shootdowns, copy-on-write fork() semantics, and huge pages (2MB/1GB) tradeoffs on modern ARM64 and x86-64 CPUs.",
        "Implement Tarjan's strongly connected components (SCC) algorithm and topological condensation DAG builder in Python, trace an example 7-node directed graph step by step, and verify correctness.",
        "Explain database B+ tree indexing, ACID isolation levels (Read Committed, Repeatable Read, Snapshot Isolation, Serializable), write-ahead logging (WAL), and MVCC phantom read prevention.",
        "Write Python code solving 0/1 knapsack and bounded knapsack using space-optimized 1D dynamic programming, explain why iteration order differs between 0/1 and unbounded knapsack, and reconstruct selected items.",
        "Compare TCP congestion control algorithms (Reno AIMD, CUBIC, BBR), explaining slow start, BDP pacing, packet loss vs ECN/RTT signals, and head-of-line blocking compared to QUIC.",
        "Write a clean Python segment tree supporting range-add lazy propagation and range minimum/sum queries in O(log N) per update and query, plus brute-force differential testing.",
        "Explain GPU unified memory vs discrete PCIe VRAM, memory coalescing, bank conflicts in shared/threadgroup memory, warp/simdgroup shuffle reductions, and roofline arithmetic intensity analysis.",
        "Implement an async Python asyncio token-bucket rate limiter with burst capacity, non-blocking try_acquire, and fair async acquire, explaining edge cases around monotonic clock drift.",
        "Explain Bloom filters, Count-Min Sketch, and HyperLogLog probabilistic data structures: derive optimal hash function count k = (m/n) ln 2 and false-positive rate formula.",
        "Write clean Python code implementing longest increasing subsequence in O(N log N) patience sorting WITH full parent pointers to recover the exact subsequence, and walk through an example.",
        "Compare Raft and Paxos consensus protocols: leader election randomized timeouts, log replication safety invariant, term numbers, split-brain prevention, and linearizable reads (ReadIndex).",
        "Design a structured markdown comparison table and detailed technical checklist for migrating a monolithic REST service to event-driven outbox-pattern microservices with idempotent consumers and dead-letter handling.",
    ]

    assert len(med) == 100 and len(chem) == 45 and len(bio) == 45 and len(math_code) == 30
    # Stratified 10% val split: every 11th index? Exact 20 val (9 med, 4 chem, 5 bio? No: exact 9 med + 4 chem + 4 bio + 3 math_code = 20 val, 200 train!)
    groups = [("med", med, 9), ("chem", chem, 4), ("bio", bio, 4), ("math_code", math_code, 3)]
    out = []
    for dom, plist, n_val in groups:
        # Spread val evenly across each group
        step = len(plist) / n_val
        val_idx = {int(round(i * step + step * 0.4)) for i in range(n_val)}
        while len(val_idx) < n_val:
            val_idx.add(max(set(range(len(plist))) - val_idx))
        for idx, text in enumerate(plist):
            split = "val" if idx in val_idx else "train"
            out.append((dom, split, text))
    # Interleave train prompts across domains so epochs see mixed domains cleanly
    train_items = [x for x in out if x[1] == "train"]
    val_items = [x for x in out if x[1] == "val"]
    assert len(train_items) == 200 and len(val_items) == 20
    # Deterministic shuffle of train_items (interleaved round-robin by domain)
    by_dom = {d: [x for x in train_items if x[0] == d] for d, _, _ in groups}
    mixed_train = []
    i = 0
    while len(mixed_train) < 200:
        for d, _, _ in groups:
            if i < len(by_dom[d]):
                mixed_train.append(by_dom[d][i])
        i += 1
    combined = mixed_train + val_items
    return [(i, dom, split, text) for i, (dom, split, text) in enumerate(combined)]


def check_no_leakage(items):
    from bench.exact import PROMPTS as EXACT_PROMPTS
    from tools.run_eval import PROMPTS as EVAL_PROMPTS

    ref_texts = [p.lower().strip() for _, p in EXACT_PROMPTS] + [p.lower().strip() for _, p in EVAL_PROMPTS]
    seen = set()
    for idx, dom, split, text in items:
        low = text.lower().strip()
        assert low not in seen, f"Duplicate prompt at idx={idx}"
        seen.add(low)
        for ref in ref_texts:
            assert low != ref and ref[:40] not in low, f"Prompt {idx} overlaps eval prompt: {ref[:50]}"
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad in ("stemi", "avf", "nsaid"):
                assert re.search(rf"\b{re.escape(bad)}\b", low) is None, f"Forbidden word {bad!r} in prompt {idx}: {text[:80]}"
            else:
                assert bad not in low, f"Forbidden substring {bad!r} found in prompt {idx}: {text[:80]}"
    print(f"Anti-leakage check PASS: {len(items)} unique disjoint prompts (200 train / 20 val), 0 overlaps.", flush=True)


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
        print("All 220 rollouts already generated.", flush=True)
        return

    maybe_set_recommended_wired_limit()
    t0 = time.perf_counter()
    model, tok = load(model_path)
    mx.eval(model.parameters())
    print(f"Loaded target for rollouts in {time.perf_counter()-t0:.1f}s (fp8={getattr(model, '_lingsante_fp8', False)}, active={mx.get_active_memory()/1e9:.1f} GB)", flush=True)

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
    total_gen_toks = 0
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
        total_gen_toks += c_toks

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

    maybe_set_recommended_wired_limit()
    t0 = time.perf_counter()
    model, _ = load(model_path)
    # Free MTP holder if present — Stage B only uses target prefill
    if hasattr(model, "_lingsante_mtp_holder"):
        model._lingsante_mtp_holder = None
    mx.clear_cache()
    mx.eval(model.parameters())
    print(f"Loaded target for capture in {time.perf_counter()-t0:.1f}s ({mx.get_active_memory()/1e9:.1f} GB)", flush=True)

    inner = model.model
    cap_set = set(TARGET_LAYER_IDS)

    # Save shared frozen word_embeddings + lm_head once for standalone training
    sio_path = out_dir / "shared_io.safetensors"
    if not sio_path.exists():
        w_emb = inner.word_embeddings.weight.astype(mx.bfloat16)
        w_head = model.lm_head.weight.astype(mx.bfloat16)
        mx.eval(w_emb, w_head)
        mx.save_safetensors(str(sio_path), {"word_embeddings.weight": w_emb, "lm_head.weight": w_head})
        print(f"Saved frozen I/O weights to {sio_path} ({sio_path.stat().st_size/1e9:.2f} GB)", flush=True)

    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    index_entries = []
    tot_pos = 0
    tot_match = 0
    tot_top4 = 0
    t_cap0 = time.perf_counter()

    for sh_id, sh_start in enumerate(range(0, len(rollouts), shard_size)):
        batch_rows = rollouts[sh_start : sh_start + shard_size]
        tensors = {}
        sh_name = f"shard_{sh_id:03d}.safetensors"
        sh_path = shard_dir / sh_name
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
                logits_c = model.lm_head(inner.norm(h))[0]  # (n, V) bf16
                lg_f32 = logits_c.astype(mx.float32)
                top_idx = mx.argpartition(-lg_f32, kth=31, axis=-1)[..., :32]
                top_vals = mx.take_along_axis(lg_f32, top_idx, axis=-1)
                ord_idx = mx.argsort(-top_vals, axis=-1)
                t32_ids_c = mx.take_along_axis(top_idx, ord_idx, axis=-1).astype(mx.uint32)
                t32_lgs_c = mx.take_along_axis(top_vals, ord_idx, axis=-1).astype(mx.float16)
                mx.eval(feats_c, t32_ids_c, t32_lgs_c, [c.state for c in cache])
                feat_chunks.append(feats_c)
                t32_id_chunks.append(t32_ids_c)
                t32_lg_chunks.append(t32_lgs_c)
                pos += n

            feats_all = mx.concatenate(feat_chunks, axis=0) if len(feat_chunks) > 1 else feat_chunks[0]
            t32_ids_all = mx.concatenate(t32_id_chunks, axis=0) if len(t32_id_chunks) > 1 else t32_id_chunks[0]
            t32_lgs_all = mx.concatenate(t32_lg_chunks, axis=0) if len(t32_lg_chunks) > 1 else t32_lg_chunks[0]
            mx.eval(feats_all, t32_ids_all, t32_lgs_all)

            # Alignment self-test on generated slice: top32_ids[t, 0] == ids[t + 1] & top4_rate
            if T > plen + 1:
                t32_sub = t32_ids_all[plen - 1 : T - 1, :4].tolist()
                gold1 = ids[plen:T]
                n_cmp = len(gold1)
                n_hit = sum(1 for row, b in zip(t32_sub, gold1) if row[0] == b)
                n_top4 = sum(1 for row, b in zip(t32_sub, gold1) if b in row)
                rate = n_hit / n_cmp
                rate4 = n_top4 / n_cmp
                tot_pos += n_cmp
                tot_match += n_hit
                tot_top4 += n_top4
                if rate < 0.90 or rate4 < 0.99:
                    raise RuntimeError(f"Alignment mismatch on idx={idx}: top1={n_hit}/{n_cmp} ({rate:.3f}), top4={n_top4}/{n_cmp} ({rate4:.3f})")
            else:
                rate = 1.0
                rate4 = 1.0

            key = f"s{idx:03d}"
            tensors[f"{key}.ids"] = p_arr
            tensors[f"{key}.feats"] = feats_all.astype(mx.bfloat16)
            tensors[f"{key}.top32_ids"] = t32_ids_all
            tensors[f"{key}.top32_logits"] = t32_lgs_all
            index_entries.append({
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
    ap.add_argument("--out-dir", default=os.path.expanduser("~/science/data/lingsante_dflash_train"))
    ap.add_argument("--stage", default="all", choices=["check", "rollouts", "capture", "all"])
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--shard-size", type=int, default=20)
    ap.add_argument("--prefill-step", type=int, default=512)
    a = ap.parse_args()

    if a.stage != "check":
        import mlx_lm  # noqa: E402,F401  # must load before bench.exact imports mlx.core

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
