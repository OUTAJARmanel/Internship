"""
Pipeline de Docking Moléculaire Automatisé
Outajar Mnael
"""

import os
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import logging
from datetime import datetime

#Je configure le système de logging afin de suivre l’exécution du pipeline et afficher les messages d’information ou d’erreur
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#Je définis une classe pour automatiser toutes les étapes nécessaires au docking moléculaire.
class MolecularDockingPipeline:
    def __init__(self):
        #Je définis les chemins principaux où se trouvent mes protéines, mes ligands et où je veux enregistrer mes résultats
        self.base_dir = "/Users/maneloutajar/Documents/Internship"
        self.proteins_dir = os.path.join(self.base_dir, "proteins")
        self.ligands_dir = os.path.join(self.base_dir, "drugs")
        self.output_dir = os.path.join(self.base_dir, "results_docking")
        
        #Suppression de l’ancien dossier de résultats (s’il existe) et je recrée un dossier propre pour stocker les nouveaux résultats
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        
        self.vina_exec = os.path.join(self.base_dir, "autodock_vina_1_1_2_mac_catalina_64bit/bin/vina")
        venv_path = os.environ.get('VIRTUAL_ENV', '/Users/maneloutajar/plip_env')
        self.pdb2pqr_exec = os.path.join(venv_path, "bin/pdb2pqr")
        
        #Création de dossiers de travail temporaires pour éviter d'écraser les originaux
        self.temp_dir = os.path.join(self.output_dir, "_temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.results = []
        self.processed_pairs = set()

    #Création de la fonction qui me permettra de lancer une commande système
    def run_cmd(self, cmd, timeout=120):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return result.returncode == 0, result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, "Commande échouée"
    
    #Préparation complète de la protéine (nettoyage et protonation)
    def clean_protein(self, protein_file):
        
        protein_name = Path(protein_file).stem
        clean_file = os.path.join(self.temp_dir, f"{protein_name}_clean.pdb")
        
        logger.info(f"Nettoyage protéine: {protein_name}")
        
        #Étape n°1: Suppression des éventuels ligands et des molécules d'eau
        amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                       'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
        
        try:
            with open(protein_file, 'r') as f_in, open(clean_file, 'w') as f_out:
                for line in f_in:
                    if line.startswith('ATOM') and line[17:20].strip() in amino_acids:
                        f_out.write(line)
                    elif line.startswith(('HEADER', 'TITLE', 'MODEL', 'ENDMDL', 'END')):
                        f_out.write(line)
        except:
            shutil.copy2(protein_file, clean_file)
        
        #Étape n°2: Ajout d'hydrogènes avec pdb2pqr si disponible
        if os.path.exists(self.pdb2pqr_exec):
            pqr_file = clean_file.replace('.pdb', '.pqr')
            cmd = [self.pdb2pqr_exec, "--ff=AMBER", "--with-ph=7.0", "--drop-water", clean_file, pqr_file]
            success, _ = self.run_cmd(cmd)
            
            if success:
                try:
                    #Conversion du fichier pqr en format pdb 
                    with open(pqr_file, 'r') as f_in, open(clean_file, 'w') as f_out:
                        for line in f_in:
                            if line.startswith(('ATOM', 'HETATM')):
                                f_out.write(line[:54] + "  1.00  0.00           \n")
                            elif line.startswith(('HEADER', 'TITLE', 'END')):
                                f_out.write(line)
                    os.remove(pqr_file)
                except:
                    pass
        
        return clean_file
    
    #Préparation du ligand pour le docking
    def prepare_ligand(self, ligand_file):
        
        ligand_name = Path(ligand_file).stem
        prepared_file = os.path.join(self.temp_dir, f"{ligand_name}_H.pdb")
        
        logger.info(f"Préparation ligand: {ligand_name}")
        
        #Étape n°1: Tentative d'ajout d'hydrogènes avec ChimeraX
        script_content = f"open {ligand_file}\naddh\nsave {prepared_file} format pdb\nexit"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cxc', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        cmd = ["ChimeraX", "--nogui", "--script", script_path]
        success, _ = self.run_cmd(cmd, timeout=60)
        os.unlink(script_path)
        
        if not success:
            shutil.copy2(ligand_file, prepared_file)
        
        return prepared_file
    #Convertion du fichier pdb en format pdbqt 
    def to_pdbqt(self, pdb_file, is_receptor=True):
        
        pdbqt_file = pdb_file.replace('.pdb', '.pdbqt')
        
        if is_receptor:
            cmd = ["obabel", "-ipdb", pdb_file, "-opdbqt", "-O", pdbqt_file, "-xr"]
        else:
            cmd = ["obabel", "-ipdb", pdb_file, "-opdbqt", "-O", pdbqt_file, "--gen3d"]
        
        success, _ = self.run_cmd(cmd)
        return pdbqt_file if success else None
        
    #Calcule du centre géométrique de la protéine
    def find_center(self, protein_file):
        
        coords = []
        try:
            with open(protein_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
            
            if coords:
                center = np.mean(coords, axis=0)
                return center[0], center[1], center[2]
        except:
            pass
        return 0, 0, 0
    #Execution du docking à l'aide de Vina
    def dock_with_vina(self, receptor_pdbqt, ligand_pdbqt, pair_name):
        
        output_pdbqt = os.path.join(self.output_dir, f"{pair_name}.pdbqt")
        log_file = os.path.join(self.output_dir, f"{pair_name}.log")
        
        #Calcul du centre de la boîte
        center_x, center_y, center_z = self.find_center(receptor_pdbqt.replace('.pdbqt', '.pdb'))
        
        vina_cmd = [
            self.vina_exec, 
            "--receptor", receptor_pdbqt, 
            "--ligand", ligand_pdbqt,
            "--out", output_pdbqt, 
            "--log", log_file,
            "--center_x", str(center_x), 
            "--center_y", str(center_y), 
            "--center_z", str(center_z),
            "--size_x", "20", "--size_y", "20", "--size_z", "20",
            "--exhaustiveness", "8", "--num_modes", "10"
        ]
        
        success, error = self.run_cmd(vina_cmd, timeout=300)
        
        if success:
            #Extraction du meilleur score de docking (kcal/mol)
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if line.strip().startswith('1 '):
                            score = float(line.split()[1])
                            return score, output_pdbqt
            except:
                pass
        
        return None, None
    #Analyse les interactions avec PLIP (étape qui ne fonctionne pas car résultats absents)
    def analyze_with_plip(self, protein_file, docked_file):
        
        try:
            from plip.structure.preparation import PDBComplex
            
            #Fichier complexe temporaire
            complex_file = os.path.join(self.temp_dir, "complex.pdb")
            
            #Conversion du résultat de docking en pdb
            ligand_pdb = docked_file.replace('.pdbqt', '_lig.pdb')
            subprocess.run(["obabel", "-ipdbqt", docked_file, "-opdb", "-O", ligand_pdb], 
                         capture_output=True, timeout=30)
            
            #Fusion protéine et ligand 
            with open(complex_file, 'w') as out:
                with open(protein_file, 'r') as prot:
                    out.write(prot.read())
                if os.path.exists(ligand_pdb):
                    with open(ligand_pdb, 'r') as lig:
                        out.write(lig.read())
            
            #Analyse PLIP
            complex_struct = PDBComplex()
            complex_struct.load_pdb(complex_file)
            
            h_bonds = 0
            hydrophobic = 0
            residues = set()
            
            for ligand_id, ligand in complex_struct.ligands.items():
                interactions = complex_struct.interaction_sets[ligand_id]
                
                
                h_bonds += len(interactions.hbonds_pdon) + len(interactions.hbonds_ldon) #Liaisons hydrogène
                
                hydrophobic += len(interactions.hydrophobic_contacts) #Contacts hydrophobes
                
                #Résidus impliqués
                for hbond in interactions.hbonds_pdon + interactions.hbonds_ldon:
                    if hasattr(hbond, 'restype') and hasattr(hbond, 'resnr'):
                        residues.add(f"{hbond.restype}{hbond.resnr}")
                
                for contact in interactions.hydrophobic_contacts:
                    if hasattr(contact, 'restype') and hasattr(contact, 'resnr'):
                        residues.add(f"{contact.restype}{contact.resnr}")
            
            #Nettoyage fichiers temporaires
            for temp_file in [complex_file, ligand_pdb]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            residues_str = ", ".join(sorted(list(residues))) if residues else "Aucun"
            return h_bonds, hydrophobic, residues_str
            
        except Exception as e:
            logger.debug(f"Erreur analyse PLIP: {str(e)}")
            return 0, 0, "Erreur"
    #Traite un couple protein-ligand
    def process_docking_pair(self, protein_file, ligand_file):
        protein_name = Path(protein_file).stem
        ligand_name = Path(ligand_file).stem
        pair_name = f"{protein_name}_{ligand_name}"
        
        #Gerer et éviter les doublons
        pair_key = (protein_name, ligand_name)
        if pair_key in self.processed_pairs:
            logger.warning(f"Paire déjà traitée: {pair_name}")
            return None
        
        self.processed_pairs.add(pair_key)
        logger.info(f"Traitement: {pair_name}")
        
        #Initialisation du résultat
        result = {
            'Protéine': protein_name,
            'Ligand': ligand_name,
            'Score_Affinité': None,
            'Liaisons_H': 0,
            'Contacts_Hydrophobes': 0,
            'Résidus_Impliqués': "Aucun",
            'Statut': 'Échec'
        }
        
        try:
            #Préparation des fichiers
            clean_protein = self.clean_protein(protein_file)
            prepared_ligand = self.prepare_ligand(ligand_file)
            
            #Conversion en format pdbqt
            protein_pdbqt = self.to_pdbqt(clean_protein, is_receptor=True)
            ligand_pdbqt = self.to_pdbqt(prepared_ligand, is_receptor=False)
            
            if not protein_pdbqt or not ligand_pdbqt:
                raise Exception("Conversion pdbqt échouée")
            
            score, docked_file = self.dock_with_vina(protein_pdbqt, ligand_pdbqt, pair_name) #Docking
            
            if score is not None and docked_file:
                result['Score_Affinité'] = round(score, 2)
                
                #Analyses des interactions
                h_bonds, hydrophobic, residues = self.analyze_with_plip(clean_protein, docked_file)
                result['Liaisons_H'] = h_bonds
                result['Contacts_Hydrophobes'] = hydrophobic
                result['Résidus_Impliqués'] = residues
                result['Statut'] = 'Succès'
                
                logger.info(f"✓ {pair_name}: {score:.2f} kcal/mol")
            else:
                raise Exception("Docking échoué")
                
        except Exception as e:
            logger.error(f"✗ {pair_name}: {str(e)}")
            result['Statut'] = f'Échec: {str(e)}'
        
        return result

    def run_complete_pipeline(self):
        """Lance le pipeline complet"""
        logger.info(" DÉBUT DU PIPELINE DE DOCKING ")
        
        #Vérification de Vina
        if not os.path.exists(self.vina_exec):
            logger.error(f"AutoDock Vina introuvable: {self.vina_exec}")
            return
        
        #Collecte des fichiers
        proteins = [f for f in os.listdir(self.proteins_dir) if f.endswith('.pdb')]
        ligands = [f for f in os.listdir(self.ligands_dir) if f.endswith('.pdb')]
        
        logger.info(f"Fichiers trouvés: {len(proteins)} protéines, {len(ligands)} ligands")
        
        if not proteins or not ligands:
            logger.error("Aucun fichier pdb trouvé dans les dossiers")
            return
        
        #Traitement de toutes les combinaisons des couples
        total_pairs = len(proteins) * len(ligands)
        current = 0
        
        for protein in proteins:
            for ligand in ligands:
                current += 1
                logger.info(f"[{current}/{total_pairs}] Début du traitement")
                
                protein_path = os.path.join(self.proteins_dir, protein)
                ligand_path = os.path.join(self.ligands_dir, ligand)
                
                result = self.process_docking_pair(protein_path, ligand_path)
                
                if result is not None:
                    self.results.append(result)
        
        self.save_results() #Export des résultats
        
        shutil.rmtree(self.temp_dir, ignore_errors=True) #Nettoyage
        
        logger.info(" PIPELINE TERMINÉE ")

    def save_results(self):
        """Sauvegarde les résultats dans un fichier Excel"""
        if not self.results:
            logger.warning("Aucun résultat à sauvegarder")
            return
        
        #Création du DataFrame
        df = pd.DataFrame(self.results)
        
        #Tri par score d'affinité (meilleur en premier)
        df_sorted = df.sort_values('Score_Affinité', na_position='last')
        
        #Sauvegarde Excel
        excel_file = os.path.join(self.output_dir, "resultats_docking.xlsx")
        df_sorted.to_excel(excel_file, index=False, engine='openpyxl')
        
        logger.info(f"Résultats sauvegardées: {excel_file}")
        
        #Résumé
        successful = df_sorted[df_sorted['Statut'] == 'Succès']
        total = len(self.results)
        success_count = len(successful)
        
        print(f"\n{'='*50}")
        print(f"RÉSULTATS DU DOCKING MOLÉCULAIRE")
        print(f"{'='*50}")
        print(f"Paires traitées: {total}")
        print(f"Succès: {success_count}")
        print(f"Échecs: {total - success_count}")
        print(f"Taux de réussite: {success_count/total*100:.1f}%")
        
        if success_count > 0:
            best = successful.iloc[0]
            print(f"\nMEILLEUR RÉSULTAT:")
            print(f"Protéine: {best['Protéine']}")
            print(f"Ligand: {best['Ligand']}")
            print(f"Score: {best['Score_Affinité']:.2f} kcal/mol")
            print(f"Liaisons H: {best['Liaisons_H']}")
            print(f"Contacts hydrophobes: {best['Contacts_Hydrophobes']}")
        
        print(f"\nFichier généré: resultats_docking.xlsx")
        print(f"{'='*50}")

def main():
    """Fonction principale"""
    pipeline = MolecularDockingPipeline()
    pipeline.run_complete_pipeline()

if __name__ == "__main__":
    main()