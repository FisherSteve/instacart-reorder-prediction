-- ============================================================================
-- INSTACART REORDER PREDICTION - FEATURE ENGINEERING
-- ============================================================================
-- Als studentisches Lernprojekt bewusst ausführlich kommentiert
-- 
-- Ziel: Aus den Instacart CSV-Dateien Features für Reorder-Prediction erstellen
-- Wichtig: Strikte Trennung zwischen 'prior' (für Features) und 'train' (nur für Labels)
-- um Data Leakage zu vermeiden
-- ============================================================================

-- 1) QUELLEN EINLESEN
--  lese  alle CSV-Dateien direkt mit DuckDB ein, ohne ETL-Zwischenschritte
-- ist effizienter und reduziert die Komplexität der Pipeline

-- Bestellungen (orders.csv)
-- Enthält: order_id, user_id, eval_set, order_number, order_dow, order_hour_of_day, days_since_prior_order
CREATE OR REPLACE VIEW orders_raw AS
SELECT * FROM read_csv_auto('data/raw/orders.csv');

-- Produkte in Bestellungen - Prior Set (order_products__prior.csv)  
-- Enthält: order_id, product_id, add_to_cart_order, reordered
-- Wichtig: Nur diese Daten verwenden wir für Feature-Engineering!
CREATE OR REPLACE VIEW order_products_prior_raw AS
SELECT * FROM read_csv_auto('data/raw/order_products__prior.csv');

-- Produkte in Bestellungen - Train Set (order_products__train.csv)
-- Enthält: order_id, product_id, add_to_cart_order, reordered  
-- Wichtig: Nur für Labels verwenden, NICHT für Features!
CREATE OR REPLACE VIEW order_products_train_raw AS
SELECT * FROM read_csv_auto('data/raw/order_products__train.csv');

-- Produktkatalog (products.csv)
-- Enthält: product_id, product_name, aisle_id, department_id
CREATE OR REPLACE VIEW products_raw AS
SELECT * FROM read_csv_auto('data/raw/products.csv');

-- Gänge/Aisles (aisles.csv)
-- Enthält: aisle_id, aisle
CREATE OR REPLACE VIEW aisles_raw AS
SELECT * FROM read_csv_auto('data/raw/aisles.csv');

-- Abteilungen/Departments (departments.csv)  
-- Enthält: department_id, department
CREATE OR REPLACE VIEW departments_raw AS
SELECT * FROM read_csv_auto('data/raw/departments.csv');

-- ============================================================================
-- 2) PRIOR/TRAIN TRENNUNG
-- ============================================================================
--  trenne strikt zwischen den Datensätzen um Data Leakage zu vermeiden
-- Prior: Für Feature-Engineering verwenden
-- Train: Nur für Label-Generierung verwenden

-- Prior Orders: Alle Bestellungen aus dem 'prior' eval_set
-- verwenden für die Feature-Berechnung
CREATE OR REPLACE VIEW prior_orders AS
SELECT 
    order_id,
    user_id,
    order_number,
    order_dow,                    -- Tag der Woche (0=Sonntag)
    order_hour_of_day,           -- Stunde des Tages
    days_since_prior_order       -- Tage seit letzter Bestellung
FROM orders_raw 
WHERE eval_set = 'prior';

-- Train Orders: Alle Bestellungen aus dem 'train' eval_set  
--  verwenden wir NUR für die Label-Generierung
CREATE OR REPLACE VIEW train_orders AS
SELECT 
    order_id,
    user_id,
    order_number,
    order_dow,
    order_hour_of_day,
    days_since_prior_order
FROM orders_raw 
WHERE eval_set = 'train';

-- ============================================================================
-- 3) USER-PRODUCT INTERAKTIONS-FEATURES AUS PRIOR DATA
-- ============================================================================
-- berechne für jede User-Product Kombination verschiedene Features
-- basierend auf dem historischen Kaufverhalten (nur aus prior orders)

-- Basis: Alle User-Product Kombinationen aus Prior Orders
--  joine die prior orders mit den entsprechenden Produkten
CREATE OR REPLACE VIEW user_product_prior_base AS
SELECT 
    po.user_id,
    opp.product_id,
    po.order_id,
    po.order_number,
    opp.add_to_cart_order,       -- Position im Warenkorb
    opp.reordered,               -- War es ein Reorder? (1=ja, 0=nein)
    po.days_since_prior_order
FROM prior_orders po
JOIN order_products_prior_raw opp ON po.order_id = opp.order_id;

-- User-Product Aggregations: Historisches Kaufverhalten pro User-Product Paar
--  berechne i die wichtigsten Features für die Reorder-Prediction
CREATE OR REPLACE VIEW user_product_features AS
SELECT 
    user_id,
    product_id,
    
    -- Feature 1: Wie oft hat der User dieses Produkt gekauft?
    -- Rationale: Häufig gekaufte Produkte werden eher wieder bestellt
    COUNT(*) as times_bought,
    
    -- Feature 2: Wie oft war es ein Reorder (nicht beim ersten Kauf)?
    -- Rationale: Produkte die schon mal reordered wurden, werden eher wieder reordered
    SUM(reordered) as times_reordered,
    
    -- Feature 3: Reorder-Rate für dieses User-Product Paar
    -- Rationale: Verhältnis von Reorders zu Gesamtkäufen zeigt Loyalität
    CASE 
        WHEN COUNT(*) > 1 THEN SUM(reordered)::FLOAT / (COUNT(*) - 1)
        ELSE 0.0 
    END as user_prod_reorder_rate,
    
    -- Feature 4: Letzte Bestellnummer mit diesem Produkt
    -- Rationale: Brauchen wir für Recency-Berechnung
    MAX(order_number) as last_prior_ordnum,
    
    -- Feature 5: Durchschnittliche Position im Warenkorb
    -- Rationale: Produkte die früh in den Korb gelegt werden sind wichtiger
    AVG(add_to_cart_order::FLOAT) as avg_add_to_cart_pos,
    
    -- Feature 6: Durchschnittliche Tage zwischen Bestellungen (für diesen User)
    -- Rationale: Zeigt Kaufrhythmus des Users
    AVG(days_since_prior_order::FLOAT) as avg_days_since_prior

FROM user_product_prior_base
GROUP BY user_id, product_id;

-- ============================================================================
-- 4) RECENCY FEATURES
-- ============================================================================
-- Recency ist extrem wichtig für Reorder-Prediction!
-- Produkte die kürzlich gekauft wurden, werden eher wieder bestellt

-- User-Level: Maximale Bestellnummer pro User (= letzte Prior-Bestellung)
--  berechnen wie lange ein Produkt nicht mehr gekauft wurde
CREATE OR REPLACE VIEW user_max_prior_order AS
SELECT 
    user_id,
    MAX(order_number) as user_max_prior_ordnum
FROM prior_orders
GROUP BY user_id;

-- Recency Feature: Wie viele Bestellungen ist es her seit letztem Kauf?
-- Je kleiner die Zahl, desto aktueller das Produkt
CREATE OR REPLACE VIEW user_product_recency AS
SELECT 
    upf.user_id,
    upf.product_id,
    upf.times_bought,
    upf.times_reordered,
    upf.user_prod_reorder_rate,
    upf.last_prior_ordnum,
    upf.avg_add_to_cart_pos,
    upf.avg_days_since_prior,
    
    -- Feature 7: Orders since last purchase
    -- Rationale: Je mehr Bestellungen ohne dieses Produkt, desto unwahrscheinlicher ein Reorder
    (umpo.user_max_prior_ordnum - upf.last_prior_ordnum) as orders_since_last

FROM user_product_features upf
JOIN user_max_prior_order umpo ON upf.user_id = umpo.user_id;

-- ============================================================================
-- 5) PRODUCT POPULARITY FEATURES
-- ============================================================================
-- Manche Produkte sind generell beliebter als andere
-- wichtiger Indikator für Reorder-Wahrscheinlichkeit

-- Global Product Statistics: Wie beliebt ist jedes Produkt insgesamt?
CREATE OR REPLACE VIEW product_popularity AS
SELECT 
    product_id,
    
    -- Feature 8: Wie oft wurde dieses Produkt insgesamt gekauft?
    -- Rationale: Beliebte Produkte werden eher reordered
    COUNT(*) as prod_cnt,
    
    -- Feature 9: Von wie vielen verschiedenen Usern wurde es gekauft?
    -- Rationale: Produkte mit breiter User-Base sind stabiler
    COUNT(DISTINCT user_id) as prod_users,
    
    -- Feature 10: Durchschnittliche Reorder-Rate für dieses Produkt
    -- Rationale: Manche Produkte sind generell "reorder-freundlicher"
    AVG(reordered::FLOAT) as prod_avg_reorder_rate

FROM user_product_prior_base
GROUP BY product_id;

-- ============================================================================
-- 6) CATEGORICAL FEATURES (AISLE & DEPARTMENT LOOKUPS)
-- ============================================================================
-- Produktkategorien sind wichtig: Manche Aisles/Departments haben höhere Reorder-Raten

-- Product Enrichment: Füge Aisle und Department Informationen hinzu
-- kategorische Features die das ML-Modell nutzen kann
CREATE OR REPLACE VIEW product_categories AS
SELECT 
    p.product_id,
    p.aisle_id,
    p.department_id,
    a.aisle as aisle_name,           -- Für Debugging/Verständnis
    d.department as department_name   -- Für Debugging/Verständnis
FROM products_raw p
LEFT JOIN aisles_raw a ON p.aisle_id = a.aisle_id
LEFT JOIN departments_raw d ON p.department_id = d.department_id;

-- ============================================================================
-- 7) LABEL GENERATION (NUR AUS TRAIN DATA!)
-- ============================================================================
--  generiere die Labels (y) aus den Train Orders
-- WICHTIG: Das ist der einzige Ort wo ich Train Data verwende!

-- Train Labels: Welche User-Product Paare wurden in Train Orders reordered?
CREATE OR REPLACE VIEW train_labels AS
SELECT 
    tr.user_id,
    opt.product_id,
    1 as y  -- Label: 1 = wurde reordered, 0 = wurde nicht reordered
FROM train_orders tr
JOIN order_products_train_raw opt ON tr.order_id = opt.order_id;

-- ============================================================================
-- 8) FINAL FEATURE SET ASSEMBLY
-- ============================================================================
--  füge  alle Features zusammen und erstelle das finale Dataset
-- Jede Zeile = ein User-Product Paar mit Features und Label

-- Basis: Alle User-Product Kombinationen die in Prior Orders vorkommen
-- Rationale: Wir können nur Reorders für Produkte vorhersagen die der User schon mal gekauft hat
CREATE OR REPLACE VIEW candidate_user_products AS
SELECT DISTINCT 
    user_id,
    product_id
FROM user_product_prior_base;

-- ============================================================================
-- 8) FINAL FEATURE SET ASSEMBLY
-- ============================================================================
-- füge alle Features zusammen und erstelle das finale Dataset
-- Jede Zeile = ein User-Product Paar mit Features und Label

-- Final Feature Assembly: Alle Features + Labels zusammenfügen
SELECT 
    -- Identifiers
    cup.user_id,
    cup.product_id,
    
    -- Label (0 wenn nicht in train_labels, 1 wenn drin)
    -- Das ist unser Zielvariable für die ML-Modelle
    COALESCE(tl.y, 0) as y,
    
    -- User-Product Interaction Features (aus Prior Data)
    upr.times_bought,
    upr.times_reordered, 
    upr.user_prod_reorder_rate,
    upr.last_prior_ordnum,
    upr.orders_since_last,
    upr.avg_add_to_cart_pos,
    upr.avg_days_since_prior,
    
    -- Product Popularity Features (aus Prior Data)
    pp.prod_cnt,
    pp.prod_users,
    pp.prod_avg_reorder_rate,
    
    -- Categorical Features (Aisle & Department)
    -- werden später im Python Code mit OneHotEncoder verarbeitet
    pc.aisle_id,
    pc.department_id

FROM candidate_user_products cup

-- Join User-Product Features (mit Recency)
LEFT JOIN user_product_recency upr 
    ON cup.user_id = upr.user_id AND cup.product_id = upr.product_id

-- Join Product Popularity Features  
LEFT JOIN product_popularity pp 
    ON cup.product_id = pp.product_id
    
-- Join Product Categories
LEFT JOIN product_categories pc 
    ON cup.product_id = pc.product_id

-- Join Train Labels (nur für User-Products die tatsächlich reordered wurden)
LEFT JOIN train_labels tl 
    ON cup.user_id = tl.user_id AND cup.product_id = tl.product_id

-- Filter: Nur User die auch Train Orders haben
-- Rationale: Wir können nur für User Predictions machen für die wir Labels haben
WHERE cup.user_id IN (SELECT DISTINCT user_id FROM train_orders)

-- Sortierung für bessere Lesbarkeit und Debugging
ORDER BY cup.user_id, cup.product_id;

-- ============================================================================
-- ENDE DER FEATURE ENGINEERING PIPELINE
-- ============================================================================
-- 
-- Output Schema:
-- - user_id: User Identifier
-- - product_id: Product Identifier  
-- - y: Binary Label (1=reordered, 0=not reordered)
-- - times_bought: Anzahl Käufe in Prior Orders
-- - times_reordered: Anzahl Reorders in Prior Orders
-- - user_prod_reorder_rate: Reorder Rate für dieses User-Product Paar
-- - last_prior_ordnum: Letzte Order Number mit diesem Produkt
-- - orders_since_last: Bestellungen seit letztem Kauf (Recency)
-- - avg_add_to_cart_pos: Durchschnittliche Warenkorb-Position
-- - avg_days_since_prior: Durchschnittliche Tage zwischen Bestellungen
-- - prod_cnt: Globale Produktpopularität (Anzahl Käufe)
-- - prod_users: Anzahl verschiedener User die das Produkt kauften
-- - prod_avg_reorder_rate: Durchschnittliche Reorder-Rate für das Produkt
-- - aisle_id: Produktkategorie (Gang)
-- - department_id: Produktkategorie (Abteilung)
--
-- Dieses Dataset kann direkt für ML-Training verwendet werden!
-- ============================================================================