#!/usr/bin/env python
import glob
import json
import tempfile
import logging
from tqdm import tqdm
import requests
import random
import tarfile

# ==================UTILS======================
from Bio.PDB import MMCIFParser, MMCIFIO
from Bio import Align
from colorama import Fore, Style
from io import StringIO
from typing import Mapping, Tuple, List

import configparser
import logging
import os
import time


# Custom formatter for colored logging
class ColoredFormatter(logging.Formatter):
    # Define color codes for each log level
    LEVEL_COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.WHITE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        # Get the color for the log level
        level_color = self.LEVEL_COLORS.get(record.levelno, "")
        # Format the log message
        formatted_message = super().format(record)
        # Return the message with the color added
        return f"{level_color}{formatted_message}{Style.RESET_ALL}"

# Set up logging
def setup_logger():
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)  # Set the minimum logging level

    # Create a stream handler (output to console)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    # Set the custom formatter
    formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger

def check_chains(mmcif_file):
    """Return a list of chains in a MMCIF file."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("template", mmcif_file)
    chains = []
    for model in structure:
        for chain in model:
            chains.append(chain.id)
    return chains


def extract_sequence_from_mmcif(mmcif_file):
    """Extract the sequence from a MMCIF file."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("template", mmcif_file)
    sequence = ""
    model = structure[0] # Assuming one model/chain only
    for chain in model:
        for residue in chain:
            if residue.id[0] == " ":  # Exclude heteroatoms
                sequence += residue.resname[
                    0
                ]  # Simplified to take the first letter
    return sequence

# Code from https://github.com/google-deepmind/alphafold3
# Modified since original one had query_index and hit_index swapped
def query_to_hit_mapping(query_aligned: str, template_aligned: str) -> Mapping[int, int]:
    """0-based query index to hit index mapping."""
    query_to_hit_mapping_out = {}
    hit_index = 0
    query_index = 0
    for q_char, t_char in zip(query_aligned, template_aligned):
        # Gap in the query
        if q_char == '-':
            hit_index += 1
        # Gap in the template
        elif t_char == '-':
            query_index += 1
        # Normal aligned residue, in both query and template. Add to mapping.
        else:
            query_to_hit_mapping_out[query_index] = hit_index
            query_index += 1
            hit_index += 1
    return query_to_hit_mapping_out


def align_and_map(query_seq, template_seq):
    """Align two sequences and map the indices."""
    # Perform pairwise alignment
    aligner = Align.PairwiseAligner() # Maybe its useful to add the options that alphafold uses to search templates
    alignments = aligner.align(query_seq, template_seq)
    alignment = alignments[0]  # Take the best alignment
    query_aligned, template_aligned = alignment[0], alignment[1]

    # Map the aligned sequences
    aligned_mapping = query_to_hit_mapping(query_aligned, template_aligned)
    
    query_indices = []
    template_indices = []
    for query_index, template_index in aligned_mapping.items():
        query_indices.append(query_index)
        template_indices.append(template_index)

    return query_indices, template_indices


def get_mmcif(
    cif,
    pdb_id,
    chain_id,
    start,
    end,
    tmpdir=None,
):
    """Extract a chain from a CIF file and return a new CIF string with only the specified chain, residues and metadata."""

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, cif)

    # Extract release date from the CIF file
    mmcif_dict = parser._mmcif_dict
    headers_to_keep = [
        "_entry.id",
        "_entry.title",
        "_entry.deposition_date",
        "_pdbx_audit_revision_history.revision_date",
    ]
    filtered_metadata = {
        key: mmcif_dict[key] for key in headers_to_keep if key in mmcif_dict
    }

    # Make metadata if missing
    if "_pdbx_audit_revision_history.revision_date" not in filtered_metadata:
        filtered_metadata["_pdbx_audit_revision_history.revision_date"] = time.strftime(
            "%Y-%m-%d"
        )

    
    # For multimodel templates (e.g. NMR) pick a single representative model
    if len(structure) > 1:
        for model_index in range(1, len(structure)):
            structure.detach_child(structure[model_index].get_id())

    for model in structure:
        chain_to_del = []
        for chain in model:
            if chain.id != chain_id:
                chain_to_del.append(chain.id)
                continue

        for unwanted_chain in chain_to_del:
            model.detach_child(unwanted_chain)

        for chain in model:
            res_to_del = []
            for i, res in enumerate(chain):
                rel_pos = i + 1
                if rel_pos < start or rel_pos > end or res.id[0] != " ":
                    res_to_del.append(res)

            for res in res_to_del:
                chain.detach_child(res.id)

    # Save the filtered structure to a new CIF file
    io = MMCIFIO()
    io.set_structure(structure)
    filtered_output = (
        f"{pdb_id}_{chain_id}.cif"
        if tmpdir is None
        else f"{tmpdir}/{pdb_id}_{chain_id}.cif"
    )
    io.save(filtered_output)

    # Parse the filtered structure to get the modified MMCIF with no metadata
    structure = parser.get_structure(pdb_id, filtered_output)
    mmcif_dict = parser._mmcif_dict

    # Add the filtered metadata to the MMCIF dictionary
    mmcif_dict.update(filtered_metadata)

    # Save the modified MMCIF with wanted metadata to a string
    string_io = StringIO()
    io.set_dict(mmcif_dict)
    io.save(string_io)

    os.unlink(filtered_output)

    return string_io.getvalue()


def mmseqs2_argparse_util(parser):
    parser.add_argument(
        "--templates", action="store_true", help="Include templates in the output json"
    )
    parser.add_argument(
        "--num_templates",
        type=int,
        default=20,
        help="Number of templates to include in the output json",
    )
    return parser

# =======================END UTILS===========================

TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"
)

class MMseqs2Exception(Exception):
    def __init__(self):
        msg = "MMseqs2 API is giving errors. Please confirm your input is a valid \
protein sequence. If error persists, please try again an hour later."
        super().__init__(msg)


# Does not support custom templates
# Only computes msa for protein sequences
def add_msa_to_folder(
    input_dir,
    output_dir,
    templates,
    num_templates,
):
    # Check json files in the folder
    if not os.path.isdir(input_dir):
        logger.error("The input folder does not exist")
        return
    # Get all json files
    json_filenames = glob.glob(os.path.join(input_dir, "*.json"))
    if len(json_filenames) == 0:
        logger.warning("No json files found in the input folder")
        return

    # Create output folder if it does not exist or it is empty
    if not os.path.isdir(output_dir) or not os.listdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    else:
        logger.error("The output folder is not empty")
        return

    # Get all protein sequences first. And also file index and sequence index in the file (list of tuples)
    sequences = []
    locations = [] # List of tuples containing the file a sequence corresponds to, and the sequence index in that file. 
    for file_i, filename in enumerate(json_filenames):
        with open(filename, "r") as file:
            af3_json = json.load(file)
            for seq_i, sequence in enumerate(af3_json["sequences"]):
                if "protein" in sequence:
                    locations.append((file_i, seq_i))
                    sequences.append(sequence["protein"]["sequence"])
     
    # Run mmseqs on all sequences and get their msas
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info(f'Running MMseqs2 on {len(sequences)} sequences.')
        if templates:
            a3m_lines, templates = run_mmseqs(
                sequences,
                tmpdir,
                use_templates=True,
                num_templates=num_templates,
            )
        else:
            a3m_lines = run_mmseqs(sequences, tmpdir, use_templates=False)
            templates = [[]*len(sequences)]
            
    # Add unpaired MSAs to the jsons
    for file_i, filename in enumerate(json_filenames):
        # Get sequence indices to write
        seqidx_and_writelocs = [(seq_list_i, loc[1]) for seq_list_i, loc in enumerate(locations) if loc[0] == file_i]
        if len(seqidx_and_writelocs) == 0:
            continue
        
        with open(filename, "r") as file:
            af3_json = json.load(file)
        
        for seqidx, writeloc in seqidx_and_writelocs:
            af3_json["sequences"][writeloc]["protein"]["unpairedMsa"] = a3m_lines[seqidx]
            af3_json["sequences"][writeloc]["protein"]["pairedMsa"] = a3m_lines[seqidx]
            af3_json["sequences"][writeloc]["protein"]["templates"] = templates[seqidx]

        output_json = os.path.basename(filename).replace(".json", "_mmseqs.json")
        with open(os.path.join(output_dir, output_json), "w") as file:
            json.dump(af3_json, file)

# Code from https://github.com/sokrypton/ColabFold
# Some modifications on the template search to adapt it to AlphaFold3
def run_mmseqs(
    x,
    prefix,
    use_env=True,
    use_filter=True,
    use_templates=False,
    filter=None,
    use_pairing=False,
    host_url="https://a3m.mmseqs.com",
    num_templates=20,
) -> Tuple[List[str], List[str]]:
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    def submit(seqs, mode, N=101):
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        res = requests.post(
            f"{host_url}/{submission_endpoint}", data={"q": query, "mode": mode}
        )
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def status(ID):
        res = requests.get(f"{host_url}/ticket/{ID}")
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def download(ID, path):
        res = requests.get(f"{host_url}/result/download/{ID}")
        with open(path, "wb") as out:
            out.write(res.content)

    # process input x
    seqs = [x] if isinstance(x, str) else x

    # compatibility to old option
    if filter is not None:
        use_filter = filter

    # setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    if use_pairing:
        mode = ""
        use_templates = False
        use_env = False

    # define path
    path = prefix
    if not os.path.isdir(path):
        os.mkdir(path)

    # call mmseqs2 api
    tar_gz_file = f"{path}/out.tar.gz"
    N, REDO = 101, True

    # deduplicate and keep track of order
    seqs_unique = list(set(seqs))
    Ms = [N + seqs_unique.index(seq) for seq in seqs]
    # lets do it!
    if not os.path.isfile(tar_gz_file):
        TIME_ESTIMATE = 150 * len(seqs_unique)
        with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
            while REDO:
                pbar.set_description("SUBMIT")

                # Resubmit job until it goes through
                out = submit(seqs_unique, mode, N)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                    # resubmit
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, N)

                if out["status"] == "ERROR":
                    raise MMseqs2Exception()

                if out["status"] == "MAINTENANCE":
                    raise MMseqs2Exception()

                # wait for job to finish
                ID, TIME = out["id"], 0
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
                    time.sleep(t)
                    out = status(ID)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)

                if out["status"] == "COMPLETE":
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False

                if out["status"] == "ERROR":
                    REDO = False
                    raise MMseqs2Exception()

            # Download results
            download(ID, tar_gz_file)

    # prep list of a3m files
    if use_pairing:
        a3m_files = [f"{path}/pair.a3m"]
    else:
        a3m_files = [f"{path}/uniref.a3m"]
        if use_env:
            a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # extract a3m files
    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    # gather a3m lines
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        for line in open(a3m_file, "r"):
            if len(line) > 0:
                if "\x00" in line:
                    line = line.replace("\x00", "")
                    update_M = True
                if line.startswith(">") and update_M:
                    M = int(line[1:].rstrip())
                    update_M = False
                    if M not in a3m_lines:
                        a3m_lines[M] = []
                a3m_lines[M].append(line)

    a3m_lines = ["".join(a3m_lines[n]) for n in Ms]

    # Find templates
    if use_templates:
        templates = {n: [] for n in range(101, 101 + len(seqs_unique))} # Initialize to [] for each M
        tested_pdbs = {n: [] for n in range(101, 101 + len(seqs_unique))}
        
        logger.info("Finding and preparing templates...")

        # Create a requests session to fetch mmcif with the same TCP connection
        with requests.Session() as session, tqdm(total=len(seqs_unique), bar_format=TQDM_BAR_FORMAT) as pbar:
            for line in open(f"{path}/pdb70.m8", "r"):
                p = line.rstrip().split()
                M, pdb, qid, alilen, tstart, tend = p[0], p[1], p[2], p[3], p[8], p[9]
                M = int(M)
                qid = float(qid)
                alilen = float(alilen)
                tstart = int(tstart)
                tend = int(tend)
                # Skip if there are enough templates for M
                if len(tested_pdbs[M]) >= num_templates:
                    continue
                
                # Calculate coverage
                seq_i = M - N
                coverage = alilen / len(seqs_unique[seq_i])
    
                # Use the same template filters as AF3 and only use 1 template per pdb
                # See filtering config at alphafold3/data/pipeline.py:DataPipeline.__init__
                # See how filtering is done in alphafold3/data/templates.py:_filter_hits
                
                # Template date is not being taken into account (while on alphafold3 it is)
                # Also we are not filtering out hits with unresolved residues
                pdb_id = pdb.split("_")[0]
                if (
                        qid == 1.0 and coverage >= 0.95 # Exclude if it is an almost exact copy of the input
                        or coverage < 0.1                      # Exclude if the alignment is bad
                        or alilen < 10                         # Exclude if the alignment is too short
                        or pdb_id in tested_pdbs[M]           # Do not include duplicated templates
                    ):
                        continue
                try:
                    cif_str = fetch_mmcif(
                        session, pdb_id, pdb.split("_")[1], tstart, tend, prefix
                    )
                except Exception as e:
                    continue # Fail gracefully if the mmcif cannot be fetched or is not correct.
                    
                template = {}
                template["mmcif"] = cif_str
    
                # Align template to sequence to obtain query and template indices?
                template_seq = extract_sequence_from_mmcif(StringIO(cif_str))
                query_indices, template_indices = align_and_map(seqs_unique[seq_i], template_seq)
                
                template["queryIndices"] = query_indices
                template["templateIndices"] = template_indices
                templates[M].append(template)
                tested_pdbs[M].append(pdb_id)

                # Check if the last template has been added and update progress bar
                if len(tested_pdbs[M]) == num_templates:
                    pbar.update()
        
        templates = [templates[n] for n in Ms]
        logger.info("Found templates.")
        
    return (a3m_lines, templates) if use_templates else a3m_lines

def fetch_mmcif(
    session,
    pdb_id,
    chain_id,
    start,
    end,
    tmpdir,
):
    MAX_RETRIES = 4
    DELAY = 2
    """Fetch the mmcif file for a given PDB ID and chain ID and prepare it for use in AlphaFold3"""
    pdb_id = pdb_id.lower()
    url_base = "http://www.ebi.ac.uk/pdbe-srv/view/files/"
    url = url_base + pdb_id + ".cif"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url)
            response.raise_for_status()
            break
        except Exception as e:
            logger.error(f"Attempt {attempt} retrieving {pdb_id} failed: {e}")
            if attempt == MAX_RETRIES:
                raise e
            time.sleep(DELAY)
    text = response.text

    output = os.path.join(tmpdir, pdb_id + ".cif")
    with open(output, "w") as f:
        f.write(text)

    return get_mmcif(output, pdb_id, chain_id, start, end, tmpdir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run MMseqs2 server on a folder of alphafold3 json files"
    )
    parser.add_argument("--input_dir", help="Input folder of alphafold3 json files")
    parser.add_argument("--output_dir", help="Output folder of alphafold3 json files")

    parser = mmseqs2_argparse_util(parser)
    
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger()
    
    add_msa_to_folder(
        args.input_dir,
        args.output_dir,
        args.templates,
        args.num_templates,
    )
