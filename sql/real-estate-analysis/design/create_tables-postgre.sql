
CREATE SEQUENCE commune_idcommune_seq_1_1;

CREATE TABLE Commune (
                idCommune INTEGER NOT NULL DEFAULT nextval('commune_idcommune_seq_1_1'),
                codePostal INTEGER NOT NULL,
                nomCommune VARCHAR(50) NOT NULL,
                codeDepartement VARCHAR(10) NOT NULL,
                CONSTRAINT commune_pk PRIMARY KEY (idCommune)
);


ALTER SEQUENCE commune_idcommune_seq_1_1 OWNED BY Commune.idCommune;

CREATE SEQUENCE adresse_id_adress_seq_1_1_1;

CREATE TABLE Adresse (
                idAdresse INTEGER NOT NULL DEFAULT nextval('adresse_id_adress_seq_1_1_1'),
                numeroVoie VARCHAR(10) NOT NULL,
                typeVoie VARCHAR(20) NOT NULL,
                nomVoie VARCHAR(50) NOT NULL,
                idCommune INTEGER NOT NULL,
                CONSTRAINT adresse_pk PRIMARY KEY (idAdresse)
);


ALTER SEQUENCE adresse_id_adress_seq_1_1_1 OWNED BY Adresse.idAdresse;

CREATE SEQUENCE local_idlocal_seq;

CREATE TABLE Bien_Immobilier (
                idBienImmobilier INTEGER NOT NULL DEFAULT nextval('local_idlocal_seq'),
                typeLocal VARCHAR(20) NOT NULL,
                nbrDePieces INTEGER NOT NULL,
                surfaceCarrezLot NUMERIC(10,2) NOT NULL,
                Surface_Bati NUMERIC(10,2) NOT NULL,
                Surface_Terrain NUMERIC(10,2) NOT NULL,
                idAdresse INTEGER NOT NULL,
                CONSTRAINT bien_immobilier_pk PRIMARY KEY (idBienImmobilier)
);


ALTER SEQUENCE local_idlocal_seq OWNED BY Bien_Immobilier.idBienImmobilier;

CREATE SEQUENCE transaction_idtransac_seq;

CREATE TABLE Transaction (
                idTransaction INTEGER NOT NULL DEFAULT nextval('transaction_idtransac_seq'),
                Date_Transac DATE NOT NULL,
                natureTransaction VARCHAR(20) NOT NULL,
                valeurFonciere NUMERIC(10,2) NOT NULL,
                idBienImmobilier INTEGER NOT NULL,
                CONSTRAINT transaction_pk PRIMARY KEY (idTransaction)
);


ALTER SEQUENCE transaction_idtransac_seq OWNED BY Transaction.idTransaction;

ALTER TABLE Adresse ADD CONSTRAINT commune_adresse_fk
FOREIGN KEY (idCommune)
REFERENCES Commune (idCommune)
ON DELETE NO ACTION
ON UPDATE NO ACTION
NOT DEFERRABLE;

ALTER TABLE Bien_Immobilier ADD CONSTRAINT adresse_bien_immobilier_fk
FOREIGN KEY (idAdresse)
REFERENCES Adresse (idAdresse)
ON DELETE NO ACTION
ON UPDATE NO ACTION
NOT DEFERRABLE;

ALTER TABLE Transaction ADD CONSTRAINT bien_immobilier_transaction_fk
FOREIGN KEY (idBienImmobilier)
REFERENCES Bien_Immobilier (idBienImmobilier)
ON DELETE NO ACTION
ON UPDATE NO ACTION
NOT DEFERRABLE;
