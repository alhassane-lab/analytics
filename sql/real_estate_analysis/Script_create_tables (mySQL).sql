
CREATE TABLE Commune (
                idCommune INT AUTO_INCREMENT NOT NULL,
                codePostal INT NOT NULL,
                nomCommune VARCHAR(50) NOT NULL,
                codeDepartement VARCHAR(10) NOT NULL,
                PRIMARY KEY (idCommune)
);


CREATE TABLE Adresse (
                idAdresse INT AUTO_INCREMENT NOT NULL,
                numeroVoie VARCHAR(10) NOT NULL,
                typeVoie VARCHAR(20) NOT NULL,
                nomVoie VARCHAR(50) NOT NULL,
                idCommune INT NOT NULL,
                PRIMARY KEY (idAdresse)
);


CREATE TABLE Bien_Immobilier (
                idBienImmobilier INT AUTO_INCREMENT NOT NULL,
                typeLocal VARCHAR(20) NOT NULL,
                nbrDePieces INT NOT NULL,
                surfaceCarrezLot DECIMAL(10,2) NOT NULL,
                Surface_Bati DECIMAL(10,2) NOT NULL,
                Surface_Terrain DECIMAL(10,2) NOT NULL,
                idAdresse INT NOT NULL,
                PRIMARY KEY (idBienImmobilier)
);


CREATE TABLE Transaction (
                idTransaction INT AUTO_INCREMENT NOT NULL,
                Date_Transac DATE NOT NULL,
                natureTransaction VARCHAR(20) NOT NULL,
                valeurFonciere DECIMAL(10,2) NOT NULL,
                idBienImmobilier INT NOT NULL,
                PRIMARY KEY (idTransaction)
);


ALTER TABLE Adresse ADD CONSTRAINT commune_adresse_fk
FOREIGN KEY (idCommune)
REFERENCES Commune (idCommune)
ON DELETE NO ACTION
ON UPDATE NO ACTION;

ALTER TABLE Bien_Immobilier ADD CONSTRAINT adresse_bien_immobilier_fk
FOREIGN KEY (idAdresse)
REFERENCES Adresse (idAdresse)
ON DELETE NO ACTION
ON UPDATE NO ACTION;

ALTER TABLE Transaction ADD CONSTRAINT bien_immobilier_transaction_fk
FOREIGN KEY (idBienImmobilier)
REFERENCES Bien_Immobilier (idBienImmobilier)
ON DELETE NO ACTION
ON UPDATE NO ACTION;
