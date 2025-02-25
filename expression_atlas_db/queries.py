"""Database query functions for the Expression Atlas.

This module provides functions to query and manipulate data from the Expression Atlas 
databases (PostgreSQL/SQLite and Redshift). Functions are organized into several categories:

- Basic fetching: fetch_studies, fetch_contrasts, fetch_samples, etc.
- Queue management: fetch_studyqueue, update_studyqueue, submit_studyqueue
- Expression queries: query_differentialexpression, query_samplemeasurement
- Data visualization: build_study_adata_components, build_expression_atlas_summary_dfs

All database interactions are handled through SQLAlchemy sessions passed as arguments
to these functions.
"""

from typing import List, Union, Dict, Tuple, Callable, Any, Optional, Literal
import warnings

import pandas as pd
import anndata as ad
import numpy as np
from sqlalchemy import select, func, or_

from expression_atlas_db import base
from expression_atlas_db.utils import MetaDataFetcher

warnings.filterwarnings("ignore", message="An alias is being generated .*")


def unpack_fields(
    fields: Dict[str, Any],
    keep: Optional[Union[List[str], Callable]] = lambda x: x.startswith(
        "sample_condition"
    )
    | x.startswith("sample_type"),
) -> pd.Series:
    """Unpacks json fields column into series.

    Args:
        fields (Dict[str, Any]): Dictionary stored in the fields column as json,
            needs to be unpacked to columns
        keep (Union[List[str], Callable, None]): Filter to keep specific keys in fields.
            Can be list of columns, or a function

    Returns:
        (pd.Series): Unpacked series with filtered index with keep.
    """
    if callable(keep):
        keys, values = zip(*[f for f in fields.items() if keep(f[0])])
    else:
        keys, values = zip(*[f for f in fields.items() if not keep or f[0] in keep])
    return pd.Series(values, index=keys)


def clean_dataframe_columns(
    df: pd.DataFrame,
    drop_prefixes: List[str] = ["id", "type", "study_id"],
    drop_columns: List[str] = ["fields"],
    keep_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Removes common unwanted columns from result dataframes.

    Args:
        df (pd.DataFrame): Input dataframe to clean
        drop_prefixes (List[str]): Column prefixes to drop
        drop_columns (List[str]): Specific columns to drop
        keep_columns (Optional[List[str]]): Specific columns to keep regardless of other rules

    Returns:
        cleaned_df (pd.DataFrame): Cleaned dataframe with unwanted columns removed
    """
    mask = ~df.columns.str.startswith(tuple(drop_prefixes))
    if drop_columns:
        mask &= ~df.columns.isin(drop_columns)
    if keep_columns:
        mask |= df.columns.isin(keep_columns)

    return df.loc[:, mask]


def fetch_studies(
    session: base._Session, studies: Optional[List[str]] = None, public: bool = True
) -> pd.DataFrame:
    """Queries against the study table.

    Args:
        session (base._Session): SQLAlchemy session object to the main postgres/sqlite db
        studies (Union[List[str], None]): List of internal_id studies to query against db
        public (bool): Filter for studies without public flag set

    Returns:
        studies_df (pd.DataFrame): Studies dataframe
    """
    query = session.query(base.Study)

    if studies:
        query = session.query(base.Study).filter(base.Study.internal_id.in_(studies))

    if public:
        query = session.query(base.Study).filter(base.Study.public == True)

    results = query.all()

    df = pd.DataFrame([row.__dict__ for row in results])  

    studies_df = df.drop(columns=['_sa_instance_state'], errors='ignore')

    return studies_df


def fetch_contrasts(
    session: base._Session,
    studies: Optional[List[str]] = None,
    contrasts: Optional[List[str]] = None,
    public: bool = True,
) -> pd.DataFrame:
    """Queries against the contrast table.

    Args:
        session (base._Session): SQLAlchemy session object to the main postgres/sqlite db
        studies (Union[List[str], None]): List of internal_id studies to query against db
        contrasts (Union[List[str], None]): List of contrast_names to query against db
        public (bool): Filter for studies without public flag set

    Returns:
        contrasts_df (pd.DataFrame): Contrasts dataframe
    """

    columns = list(base.Contrast.__table__.columns) + list(base.Study.__table__.columns)

    query = session.query(*columns).join(
        base.Study, base.Contrast.study_id == base.Study.id
    )

    if studies:
        query = session.query(*columns).join(
                              base.Study, base.Contrast.study_id == base.Study.id
                        ).filter(base.Study.internal_id.in_(studies))

    if contrasts:
        query = session.query(*columns).join(
                              base.Study, base.Contrast.study_id == base.Study.id
                        ).filter(base.Contrast.conrast_name.in_(contrasts))

    if public:
        query = session.query(*columns).join(
                              base.Study, base.Contrast.study_id == base.Study.id
                        ).filter(base.Study.public == True)


    results = query.all()

    #df = pd.DataFrame([row.__dict__ for row in results])  

    contrasts_df = pd.DataFrame([row._asdict() for row in results])

    #contrasts_df = pd.read_sql(query, session.bind)

    return contrasts_df


def fetch_sequenceregions(
    session: base._Session,
    sequenceregions: Optional[List[str]] = None,
    sequenceregions_type: Optional[str] = None,
    assembly_id: Optional[str] = None,
    exact_id_match: bool = True,
) -> pd.DataFrame:
    """Queries against the sequenceregion/gene/transcript tables.

    Args:
        session (base._Session): SQLAlchemy session object to the main postgres/sqlite db
        sequenceregions (Union[List[str], None]): List of transcript_ids or gene_ids to query.
            These are the text ids, not the column ids from expression_atlas_db
        sequenceregions_type (Union[str, None]): One of "transcript", "gene", or None.
            Filter query on either table or return all
        assembly_id (Union[str, None]): OrfDB assembly_id or public assembly id
        exact_id_match (bool): Whether to allow matches to sequenceregions with prefixes

    Returns:
        sequenceregions_df (pd.DataFrame): Sequenceregions dataframe
    """

    transcript_query = session.query(base.Transcript)
    gene_query = session.query(base.Gene)

    if sequenceregions:
        if exact_id_match:
            transcript_query = transcript_query.filter(
                base.Transcript.transcript_id.in_(sequenceregions)
            )
            gene_query = gene_query.filter(base.Gene.gene_id.in_(sequenceregions))
        else:
            conditions_gene = [
                base.Gene.gene_id.ilike(f"{g}%") for g in sequenceregions
            ]
            conditions_transcript = [
                base.Transcript.transcript_id.ilike(f"{t}%") for t in sequenceregions
            ]
            transcript_query = transcript_query.filter(or_(*conditions_transcript))
            gene_query = gene_query.filter(or_(*conditions_gene))

    if assembly_id:
        transcript_query = transcript_query.filter(
            base.Transcript.assembly_id == assembly_id
        )
        gene_query = gene_query.filter(base.Gene.assembly_id == assembly_id)

    transcript_results = transcript_query.all()
    gene_results = gene_query.all()
    
    transcript_sequenceregions_df = pd.DataFrame([row.__dict__ for row in transcript_results])
    if not transcript_sequenceregions_df.empty:
        transcript_sequenceregions_df.drop("_sa_instance_state", axis=1, inplace=True, errors='ignore')
        transcript_sequenceregions_df.drop("gene_id", inplace=True, axis=1)
    
    gene_sequenceregions_df = pd.DataFrame([row.__dict__ for row in gene_results])
    if not gene_sequenceregions_df.empty:
        gene_sequenceregions_df.drop("_sa_instance_state", axis=1, inplace=True, errors='ignore')

    sequenceregions_df = pd.concat(
        [transcript_sequenceregions_df, gene_sequenceregions_df], axis=0
    )
    sequenceregions_df.set_index("id", inplace=True)

    if sequenceregions_type == "transcript":
        sequenceregions_df = sequenceregions_df[
            ~sequenceregions_df["transcript_id"].isna()
        ]
    elif sequenceregions_type == "gene":
        sequenceregions_df = sequenceregions_df[~sequenceregions_df["gene_id"].isna()]

    return sequenceregions_df


def fetch_samplecontrasts(
    session: base._Session,
    studies: Optional[List[str]] = None,
    contrasts: Optional[List[str]] = None,
    keep_fields: Optional[Union[List[str], Callable]] = lambda x: x.startswith(
        "sample_condition"
    )
    | x.startswith("sample_type"),
    public: bool = True,
) -> pd.DataFrame:
    """Queries against the samplecontrast table and merges with samples used to create the contrast.

    Args:
        session (base._Session): SQLAlchemy session object to the main postgres/sqlite db
        studies (Union[List[str], None]): List of internal_id studies to query against db
        contrasts (Union[List[str], None]): List of contrast_names to query against db
        keep_fields (Union[List[str], Callable, None]): Filter to keep specific fields columns.
            Can be list of columns or a function
        public (bool): Filter for studies without public flag set

    Returns:
        samplecontrast_df (pd.DataFrame): Sample contrasts dataframe
    """

    query = session.query(
        base.Contrast,
        base.Study.internal_id,
        base.SampleContrast.contrast_side,
        base.Sample
    ).join(
        base.Study, base.Study.id == base.Contrast.study_id
    ).join(
        base.SampleContrast, base.Contrast.id == base.SampleContrast.contrast_id
    ).join(
        base.Sample, base.SampleContrast.sample_id == base.Sample.id
    )
    
    if studies:
        query = query.filter(base.Study.internal_id.in_(studies))
    if public:
        query = query.filter(base.Study.public == True)
    if contrasts:
        query = query.filter(base.Contrast.contrast_name.in_(contrasts))

    results = query.all()
    
    # Convert results to dataframe
    rows = []
    for row in results:
        contrast, internal_id, contrast_side, sample = row
        row_dict = {**contrast.__dict__, **sample.__dict__}
        row_dict["internal_id"] = internal_id
        row_dict["contrast_side"] = contrast_side
        rows.append(row_dict)
    
    samplecontrasts_df = pd.DataFrame(rows)
    samplecontrasts_df.drop("_sa_instance_state", axis=1, inplace=True, errors='ignore')

    samplecontrasts_df = pd.concat(
        [
            samplecontrasts_df,
            samplecontrasts_df["fields"].apply(lambda x: unpack_fields(x, keep_fields)),
        ],
        axis=1,
    )

    return clean_dataframe_columns(samplecontrasts_df)


def fetch_samples(
    session: base._Session,
    studies: Union[List[str], None] = None,
    keep_fields: Union[List[str], Callable, None] = lambda x: x.startswith(
        "sample_condition"
    ),
    public: bool = True,
) -> pd.DataFrame:
    """Queries against the samples table.

    Args:
        session (base._Session): SQLAlchemy session object to the main postgres/sqlite db
        studies (Union[List[str], None]): List of internal_id studies to query against db
        keep_fields (Union[List[str], Callable, None]): Filter to keep specific fields columns.
            Can be list of columns or a function
        public (bool): Filter for studies without public flag set

    Returns:
        samples_df (pd.DataFrame): Samples dataframe
    """
    query = session.query(base.Sample, base.Study.internal_id).join(
        base.Study, base.Study.id == base.Sample.study_id
    )
    
    if studies:
        query = query.filter(base.Study.internal_id.in_(studies))
    if public:
        query = query.filter(base.Study.public == True)

    results = query.all()
    
    # Convert results to dataframe
    rows = []
    for row in results:
        sample, internal_id = row
        row_dict = {**sample.__dict__}
        row_dict["internal_id"] = internal_id
        rows.append(row_dict)
    
    samples_df = pd.DataFrame(rows)
    samples_df.drop("_sa_instance_state", axis=1, inplace=True, errors='ignore')
    
    samples_df = pd.concat(
        [
            samples_df,
            samples_df["fields"].apply(lambda x: unpack_fields(x, keep_fields)),
        ],
        axis=1,
    )

    return clean_dataframe_columns(samples_df)


def fetch_studyqueue(
    session: base._Session,
    public: bool = True,
) -> pd.DataFrame:
    """Queries against the studyqueue table.

    Args:
        session (base._Session): SQLAlchemy session object to the main postgres/sqlite db
        public (bool): Filter for studies without public flag set

    Returns:
        studyqueues_df (pd.DataFrame): Study queue dataframe
    """
    query = session.query(
        base.StudyQueue,
        base.Study.id.label("study_id"),
        base.Study.public.label("data_released"),
    ).join(
        base.Study,
        base.StudyQueue.study_id == base.Study.id,
        isouter=True,
    )
    
    if public:
        query = query.filter(base.StudyQueue.public == True)

    results = query.all()
    
    # Convert results to dataframe
    rows = []
    for row in results:
        studyqueue, study_id, data_released = row
        row_dict = {**studyqueue.__dict__}
        row_dict["study_id"] = study_id
        row_dict["data_released"] = data_released
        rows.append(row_dict)
    
    studyqueues_df = pd.DataFrame(rows)
    studyqueues_df.drop("_sa_instance_state", axis=1, inplace=True, errors='ignore')
    
    return studyqueues_df


def update_studyqueue(
    session: base._Session,
    update_rows: pd.DataFrame,
    update_columns: List[str] = [
        "tissue",
        "disease",
        "contrast",
        "category",
        "status",
        "technology",
        "quality",
        "priority",
        "comments",
    ],
) -> pd.DataFrame:
    """Update studyqueue table with metadata from editable dataframe.

    Args:
        session (base._Session): SQLAlchemy session object to the main postgres/sqlite db
        update_rows (pd.DataFrame): Dataframe of rows to be updated in studyqueue
        update_columns (List[str]): List of column names to restrict update to

    Returns:
        updated_df (pd.DataFrame): Updated rows in dataframe
    """
    for i, r in update_rows.iterrows():
        sq = (
            session.query(base.StudyQueue).filter(base.StudyQueue.id == r["id"]).first()
        )
        for c in update_columns:
            setattr(sq, c, r.get(c))
        if r.get("delete", False) == True:
            setattr(sq, "public", False)
    session.commit()
    return pd.read_sql(
        select(base.StudyQueue).filter(
            base.StudyQueue.id.in_(update_rows["id"].tolist())
        ),
        con=session.bind,
    )


def submit_studyqueue(
    session: base._Session,
    srp_id: str,
    category: str,
    technology: str,
    disease: str,
    tissue: str,
    contrast: str,
    requestor: str,
    priority: str,
    comments: str,
    geo_id: Optional[str] = None,
) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
    """Submit a study to the queue table.

    Args:
        session (base._Session): SQLAlchemy session object to the main postgres/sqlite db
        srp_id (str): SRP project id
        category (str): Reason for requesting dataset
        technology (str): Technology type for queue. Must be one of {'BULK', '10X-3', '10X-5', 'SMART-SEQ'}
        disease (str): Disease(s) for requested dataset
        tissue (str): Tissues(s) for requested dataset
        contrast (str): Contrast(s) for requested dataset
        requestor (str): Requestor(s) for requested dataset
        priority (str): Priority level for requested dataset
        comments (str): General comments for requested dataset
        geo_id (Union[str, None]): GEO ID for project

    Returns:
        (Tuple[bool, Optional[pd.DataFrame], Optional[str]]): Study exists flag, results dataframe, message string
    """
    if technology not in (
        "BULK",
        "10X-3",
        "10X-5",
        "SMART-SEQ",
        "OTHER",
    ):
        return (
            False,
            f"Technology must be one of {'BULK', '10X-3', '10X-5', 'SMART-SEQ', 'OTHER'}, is: {srp_id}.",
        )
    studyqueue = (
        session.query(base.StudyQueue)
        .filter(
            (base.StudyQueue.srp_id == srp_id)
            | ((base.StudyQueue.geo_id == geo_id) & (base.StudyQueue.geo_id != None))
        )
        .first()
    )

    if not studyqueue:
        try:
            meta = MetaDataFetcher(srp_id, [])
        except Exception as e:
            return (False, e)
        if (meta.srp_id and geo_id and meta.geo_id == geo_id) or (
            not geo_id and meta.srp_id
        ):
            studyqueue = base.StudyQueue(
                internal_id=srp_id,
                geo_id=geo_id,
                srp_id=srp_id,
                pmid=meta.pmids,
                title=meta.project_title,
                description=meta.project_summary,
                public=True,
                processed=False,
                status="QUEUED",
                technology=technology,
                category=category,
                disease=disease,
                tissue=tissue,
                requestor=requestor,
                contrast=contrast,
                priority=priority,
                comments=comments,
            )
            session.add(studyqueue)
            return (
                False,
                pd.DataFrame(
                    [
                        {
                            c.name: getattr(studyqueue, c.name, None)
                            for c in base.StudyQueue.__table__.columns
                        }
                    ],
                ),
            )
        elif geo_id and meta.geo_id != geo_id:
            return (
                False,
                f"Submitted geo id and geo id associated with {srp_id} don't match.",
            )
        else:
            return (False, f"Cannot request metadata for {srp_id}.")

    else:
        return (
            True,
            pd.read_sql(
                select(base.StudyQueue).filter(base.StudyQueue.id == studyqueue.id),
                session.bind,
            ),
        )


def query_differentialexpression(
    session: base._Session,
    session_redshift: base._Session,
    studies: Optional[List[str]] = None,
    contrasts: Optional[List[str]] = None,
    sequenceregions: Optional[List[str]] = None,
    sequenceregions_type: Optional[str] = None,
    log10_padj_threshold: Optional[float] = np.log10(0.05),
    log2_fc_threshold: Optional[float] = np.log2(2.0),
    mean_threshold: Optional[float] = 4.0,
    public: bool = True,
    exact_id_match: bool = True,
) -> pd.DataFrame:
    """Queries differential expression data from both databases.

    First queries contrast and sequenceregion tables in postgres/sqlite db,
    then uses contrast_ids and sample_ids to query differentialexpression table in redshift.

    Args:
        session (base._Session): SQLAlchemy session object to the main postgres/sqlite db
        session_redshift (base._Session): SQLAlchemy session object to redshift db
        studies (Union[List[str], None]): List of internal_id studies to query against db
        contrasts (Union[List[str], None]): List of contrast_names to query against db
        sequenceregions (Union[List[str], None]): List of transcript_ids or gene_ids to query.
            These are the text ids, not the column ids
        sequenceregions_type (Union[str, None]): One of "transcript", "gene", or None.
            Filter query on either table or return all
        log10_padj_threshold (Union[float, None]): Keep rows log10_padj < threshold
        log2_fc_threshold (Union[float, None]): Keep rows abs(log2(fc)) > threshold
        mean_threshold (Union[float, None]): Keep rows mean(normed_transformed_count) in case or control
        public (bool): Filter for studies without public flag set
        exact_id_match (bool): Whether to allow matches to sequenceregions with prefixes

    Returns:
        differentialexpression_df (pd.DataFrame): Differential expression dataframe
    """

    log10_padj_threshold = float(np.log10(0.05))
    log2_fc_threshold = float(np.log2(2.0))

    # Query for contrasts and studies
    studies_query = session.query(
        base.Contrast,
        base.Study.internal_id
    ).join(
        base.Study, base.Contrast.study_id == base.Study.id
    )

    if studies:
        studies_query = studies_query.filter(base.Study.internal_id.in_(studies))
    if public:
        studies_query = studies_query.filter(base.Study.public == True)
    if contrasts:
        studies_query = studies_query.filter(base.Contrast.contrast_name.in_(contrasts))

    results = studies_query.all()
    
    # Convert results to dataframe
    rows = []
    for row in results:
        contrast, internal_id = row
        row_dict = {**contrast.__dict__}
        row_dict["internal_id"] = internal_id
        rows.append(row_dict)
    
    studies_df = pd.DataFrame(rows)
    studies_df.drop("_sa_instance_state", axis=1, inplace=True, errors='ignore')
    studies_df.set_index("id", inplace=True)

    sequenceregions_df = fetch_sequenceregions(
        session,
        sequenceregions=sequenceregions,
        sequenceregions_type=sequenceregions_type,
        exact_id_match=exact_id_match,
    )

    differentialexpression_query = session_redshift.query(base.DifferentialExpression).filter(
        base.DifferentialExpression.contrast_id.in_(studies_df.index)
    )

    if sequenceregions:
        differentialexpression_query = differentialexpression_query.filter(
            base.DifferentialExpression.sequenceregion_id.in_(sequenceregions_df.index)
        )

    if log10_padj_threshold:
        differentialexpression_query = differentialexpression_query.filter(
            base.DifferentialExpression.log10_padj <= log10_padj_threshold
        )

    if mean_threshold:
        differentialexpression_query = differentialexpression_query.filter(
            (base.DifferentialExpression.case_mean >= mean_threshold)
            | (base.DifferentialExpression.control_mean >= mean_threshold)
        )

    if log2_fc_threshold:
        differentialexpression_query = differentialexpression_query.filter(
            func.abs(base.DifferentialExpression.log2foldchange) >= log2_fc_threshold
        )

    results = differentialexpression_query.all()
    differentialexpression_df = pd.DataFrame([row.__dict__ for row in results])
    differentialexpression_df.drop("_sa_instance_state", axis=1, inplace=True, errors='ignore')
    
    differentialexpression_df = differentialexpression_df.merge(
        studies_df[["internal_id", "contrast_name"]],
        left_on="contrast_id",
        right_index=True,
        how="left",
    ).merge(
        sequenceregions_df[["transcript_id", "gene_id", "type"]],
        left_on="sequenceregion_id",
        right_index=True,
    )

    return clean_dataframe_columns(
        differentialexpression_df,
        drop_prefixes=["id", "contrast_id", "sequenceregion_id", "type"],
        keep_columns=["transcript_id", "gene_id"],
    )


def query_samplemeasurement(
    session: base._Session,
    session_redshift: base._Session,
    studies: Optional[List[str]] = None,
    contrasts: Optional[List[str]] = None,
    samples: Optional[List[str]] = None,
    sequenceregions: Optional[List[str]] = None,
    sequenceregions_type: Optional[str] = None,
    public: bool = True,
    exact_id_match: bool = True,
) -> pd.DataFrame:
    """Queries sample measurements from both databases.

    First queries contrast and sequenceregion tables in postgres/sqlite db,
    then uses contrast_ids and sample_ids to query samplemeasurement table in redshift.

    Args:
        session (base._Session): SQLAlchemy session object to the main postgres/sqlite db
        session_redshift (base._Session): SQLAlchemy session object to redshift db
        studies (Union[List[str], None]): List of internal_id studies to query against db
        contrasts (Union[List[str], None]): List of contrast_names to query against db
        samples (Union[List[str], None]): List of sample ids to query
        sequenceregions (Union[List[str], None]): List of transcript_ids or gene_ids to query.
            These are the text ids, not the column ids
        sequenceregions_type (Union[str, None]): One of "transcript", "gene", or None.
            Filter query on either table or return all
        public (bool): Filter for studies without public flag set
        exact_id_match (bool): Whether to allow matches to sequenceregions with prefixes

    Returns:
        samplemeasurement_df (pd.DataFrame): Sample measurements dataframe
    """
    samples_query = session.query(base.Sample, base.Study.internal_id).join(
        base.Study, base.Sample.study_id == base.Study.id
    )
    if studies:
        samples_query = samples_query.filter(base.Study.internal_id.in_(studies))

    if public:
        samples_query = samples_query.filter(base.Study.public == True)

    if samples:
        samples_query = samples_query.filter(base.Sample.srx_id.in_(samples))

    if contrasts:
        # This is more complex with ORM style, we need to handle it differently
        contrast_query = session.query(
            base.Study.internal_id,
            base.Contrast.contrast_name,
            base.SampleContrast.contrast_side,
            base.SampleContrast.sample_id
        ).join(
            base.Contrast, base.Contrast.study_id == base.Study.id
        ).join(
            base.SampleContrast, base.Contrast.id == base.SampleContrast.contrast_id
        )
        
        if studies:
            contrast_query = contrast_query.filter(base.Study.internal_id.in_(studies))
        if contrasts:
            contrast_query = contrast_query.filter(base.Contrast.contrast_name.in_(contrasts))
            
        contrast_results = contrast_query.all()
        contrast_rows = []
        for row in contrast_results:
            internal_id, contrast_name, contrast_side, sample_id = row
            contrast_rows.append({
                "internal_id": internal_id,
                "contrast_name": contrast_name,
                "contrast_side": contrast_side,
                "sample_id": sample_id
            })
        contrast_df = pd.DataFrame(contrast_rows)
        
        # Get sample results
        sample_results = samples_query.all()
        sample_rows = []
        for row in sample_results:
            sample, internal_id = row
            sample_rows.append({
                "id": sample.id,
                "internal_id": internal_id,
                "srx_id": sample.srx_id,
                "atlas_group": sample.atlas_group,
                "fields": sample.fields
            })
        samples_df = pd.DataFrame(sample_rows)
        
        # Merge the two dataframes
        samples_df = pd.merge(
            samples_df,
            contrast_df,
            left_on=["id", "internal_id"],
            right_on=["sample_id", "internal_id"],
            how="inner"
        )
    else:
        # Get sample results without contrast filtering
        sample_results = samples_query.all()
        sample_rows = []
        for row in sample_results:
            sample, internal_id = row
            sample_rows.append({
                "id": sample.id,
                "internal_id": internal_id,
                "srx_id": sample.srx_id,
                "atlas_group": sample.atlas_group,
                "fields": sample.fields
            })
        samples_df = pd.DataFrame(sample_rows)
    
    samples_df.set_index("id", inplace=True)

    sequenceregions_df = fetch_sequenceregions(
        session,
        sequenceregions=sequenceregions,
        sequenceregions_type=sequenceregions_type,
        exact_id_match=exact_id_match,
    )

    samplemeasurement_query = session_redshift.query(base.SampleMeasurement).filter(
        base.SampleMeasurement.sample_id.in_(samples_df.index)
    )

    if sequenceregions:
        samplemeasurement_query = samplemeasurement_query.filter(
            base.SampleMeasurement.sequenceregion_id.in_(sequenceregions_df.index)
        )

    results = samplemeasurement_query.all()
    samplemeasurement_df = pd.DataFrame([row.__dict__ for row in results])
    samplemeasurement_df.drop("_sa_instance_state", axis=1, inplace=True, errors='ignore')
    
    # Add fields unpacking to samples_df
    if "fields" in samples_df.columns:
        unpacked_fields = samples_df["fields"].apply(
            lambda x: unpack_fields(
                x,
                lambda x: x in (
                    "sample_condition_1",
                    "sample_condition_2",
                    "sample_type_1",
                    "sample_type_2",
                ),
            )
        )
        samples_df = pd.concat([samples_df, unpacked_fields], axis=1)
    
    # Prepare columns for merge
    sample_columns = ["internal_id", "srx_id", "atlas_group"]
    if "contrast_name" in samples_df.columns:
        sample_columns.extend(["contrast_name", "contrast_side"])
    
    # Add fields column if it exists
    if "fields" in samples_df.columns:
        sample_columns.append("fields")
    
    # Add condition and type columns if they exist
    for col in samples_df.columns:
        if col.startswith(("sample_condition_", "sample_type_")):
            sample_columns.append(col)
    
    samplemeasurement_df = samplemeasurement_df.merge(
        samples_df[sample_columns],
        left_on="sample_id",
        right_index=True,
        how="left",
    ).merge(
        sequenceregions_df[["transcript_id", "gene_id", "type"]],
        left_on="sequenceregion_id",
        right_index=True,
    )

    return clean_dataframe_columns(
        samplemeasurement_df, keep_columns=["transcript_id", "gene_id", "srx_id"]
    )


def build_contrast_metatable(
    session: base._Session,
    studies: Union[List[str], None] = None,
    contrasts: Union[List[str], None] = None,
    public: bool = True,
) -> pd.DataFrame:
    """Helper function to assemble the dashboard contrast metatable. Calls fetch_samplecontrasts
    and coerces table to format required by dashboard.

    Args:
        session (base._Session): SQLAlchemy session object to the main postgres/sqlite db.
        studies (Union[List[str],None]): List of internal_id studies to query against db.
        contrasts (Union[List[str],None]): List of contrast_names to query against db.
        public (bool): Filter for studies without public flag set.
    Returns:
        samplecontrast_df (pd.DataFrame): dashboard contrast metadata dataframe.
    """
    samplecontrast_df = fetch_samplecontrasts(
        session,
        studies=studies,
        contrasts=contrasts,
        public=public,
    ).fillna("")

    samplecontrast_df.drop(
        [
            "sample_condition_key",
            "left_condition_level",
            "right_condition_level",
            "left_condition_display",
            "right_condition_display",
            "atlas_group",
        ],
        axis=1,
        inplace=True,
    )

    samplecontrast_df = samplecontrast_df.astype(str)

    samplecontrast_df.insert(
        1,
        "sample_n",
        samplecontrast_df.groupby(["internal_id", "contrast_name", "contrast_side"])[
            "srx_id"
        ]
        .transform(len)
        .astype(str),
    )
    samplecontrast_df.drop("srx_id", inplace=True, axis=1)
    samplecontrast_df = samplecontrast_df.groupby(
        ["internal_id", "contrast_name", "contrast_side"],
    ).agg(lambda x: ",".join(x.unique()))
    for c in samplecontrast_df.columns[
        samplecontrast_df.columns.str.startswith("sample")
    ]:
        samplecontrast_df[f"left_{c}"] = samplecontrast_df.loc[
            samplecontrast_df.index.get_level_values("contrast_side") == "left"
        ][c]
        samplecontrast_df[f"right_{c}"] = samplecontrast_df.loc[
            samplecontrast_df.index.get_level_values("contrast_side") == "right"
        ][c]

    samplecontrast_df.reset_index(inplace=True, drop=False)
    samplecontrast_df.drop(
        samplecontrast_df.columns[
            samplecontrast_df.columns.str.startswith("sample")
            | samplecontrast_df.columns.str.startswith("created_at")
            | samplecontrast_df.columns.str.startswith("alembic_id")
        ].tolist()
        + ["contrast_side"],
        inplace=True,
        axis=1,
    )
    samplecontrast_df.index = samplecontrast_df.apply(
        lambda x: f"{x['internal_id']} -- {x['contrast_name']}", axis=1
    )
    samplecontrast_df = (
        samplecontrast_df.groupby(
            [samplecontrast_df.index, "internal_id", "contrast_name"]
        )
        .agg(sum)
        .reset_index(["internal_id", "contrast_name"], drop=False)
    )
    return samplecontrast_df


def query_percentile_group(
    session: base._Session,
    session_redshift: base._Session,
    sample_ids: List[int],
    percentile_levels: List[float] = [0.5, 0.75, 0.90],
    sequenceregion_ids: Optional[List[int]] = None,
    aggregate_column: str = "tpm",
) -> pd.DataFrame:
    """Queries expression percentiles for sample groups.

    Args:
        session (base._Session): SQLAlchemy session object to the main postgres/sqlite db
        session_redshift (base._Session): SQLAlchemy session object to redshift db
        sample_ids (List[int]): List of sample_ids to query against db
        percentile_levels (List[float]): List of percentile levels to query
        sequenceregion_ids (Union[List[int], None]): List of optional sequenceregion_ids to query
        aggregate_column (str): Column to aggregate percentiles over

    Returns:
        percentile_df (pd.DataFrame): Dataframe with sequenceregion_ids and aggregated expression percentiles
    """

    if aggregate_column not in vars(base.SampleMeasurement).keys():
        raise KeyError("Aggregate_column does not exist in samplemeasurement table.")

    # This function requires raw SQL for the percentile calculations
    # We'll need to use a more direct approach with SQLAlchemy core
    from sqlalchemy.sql import text
    
    # Build the SQL query manually
    percentile_cols = ", ".join([
        f"PERCENTILE_DISC({i}) WITHIN GROUP (ORDER BY {aggregate_column}) OVER (PARTITION BY sequenceregion_id) AS perc_{i}"
        for i in percentile_levels
    ])
    
    sql = f"""
    SELECT DISTINCT sequenceregion_id, {percentile_cols}
    FROM sample_measurement
    WHERE sample_id IN :sample_ids
    """
    
    params = {"sample_ids": tuple(sample_ids)}
    
    if sequenceregion_ids:
        sql += " AND sequenceregion_id IN :sequenceregion_ids"
        params["sequenceregion_ids"] = tuple(sequenceregion_ids)
    
    # Execute the raw SQL
    result = session_redshift.execute(text(sql), params)
    
    # Convert to DataFrame
    percentile_df = pd.DataFrame([dict(row) for row in result])
    
    return percentile_df


def build_study_adata_components(
    session: base._Session,
    session_redshift: base._Session,
    studies: List[str],
    sequenceregions_type: Literal["gene", "transcript"],
    return_adata: bool = False,
    keep_fields: List[str] = [
        "sample_condition_1",
        "sample_condition_2",
        "sample_type_1",
        "sample_type_2",
    ],
) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], ad.AnnData]:
    """Builds AnnData object or component dataframes for studies.

    Args:
        session (base._Session): SQLAlchemy session object to the main postgres/sqlite db
        session_redshift (base._Session): SQLAlchemy session object to redshift db
        studies (List[str]): List of internal_ids to query against db
        sequenceregions_type (Literal["gene", "transcript"]): Filter var by gene or transcript
        return_adata (bool): Convert to adata or return long df, obs df, var df
        keep_fields (List[str]): Keys to unpack out of fields column

    Returns:
        result (Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], ad.AnnData]):
            Either (long_df, obs_df, var_df) tuple or AnnData object
    """
    samples_query = session.query(base.Sample, base.Study.internal_id).join(
        base.Study, base.Sample.study_id == base.Study.id
    ).filter(base.Study.internal_id.in_(studies))

    sample_results = samples_query.all()
    sample_rows = []
    for row in sample_results:
        sample, internal_id = row
        sample_rows.append({
            **sample.__dict__,
            "internal_id": internal_id
        })
    
    samples_df = pd.DataFrame(sample_rows)
    samples_df.drop("_sa_instance_state", axis=1, inplace=True, errors='ignore')
    samples_df.drop("id_1", axis=1, errors='ignore')
    
    samples_df = pd.concat(
        [
            samples_df,
            samples_df["fields"].apply(
                lambda x: unpack_fields(
                    x,
                    lambda x: x in keep_fields,
                )
            ),
        ],
        axis=1,
    )
    samples_df = samples_df.sort_values("id").set_index("id")

    sequenceregions_df = (
        fetch_sequenceregions(
            session,
            sequenceregions_type=sequenceregions_type,
        )
        .drop(["assembly_id", "id_1"], axis=1, errors='ignore')
        .sort_index()
    )

    if sequenceregions_type == "transcript":
        sequenceregions_df.drop("gene_id", inplace=True, axis=1, errors='ignore')
    else:
        sequenceregions_df.drop("transcript_id", inplace=True, axis=1, errors='ignore')

    samplemeasurement_query = session_redshift.query(base.SampleMeasurement).filter(
        base.SampleMeasurement.sample_id.in_(samples_df.index)
    )

    samplemeasurement_results = samplemeasurement_query.all()
    samplemeasurement_rows = []
    for row in samplemeasurement_results:
        samplemeasurement_rows.append(row.__dict__)
    
    samplemeasurement_df = pd.DataFrame(samplemeasurement_rows)
    samplemeasurement_df.drop("_sa_instance_state", axis=1, inplace=True, errors='ignore')
    samplemeasurement_df.drop("id", axis=1, errors='ignore')
    
    samplemeasurement_df = samplemeasurement_df.loc[
        samplemeasurement_df["sequenceregion_id"].isin(sequenceregions_df.index)
    ].reset_index(drop=True)

    if return_adata:
        X_layers = samplemeasurement_df.pivot(
            index="sample_id",
            columns="sequenceregion_id",
            values=["counts", "tpm"],
        ).fillna(0.0)
        adata = ad.AnnData(
            X=X_layers["counts"],
            var=sequenceregions_df.loc[X_layers["counts"].columns],
            obs=samples_df.loc[X_layers.index],
            layers={"tpm": X_layers["tpm"]},
        )
        return adata

    return samplemeasurement_df, samples_df, sequenceregions_df


def build_expression_atlas_summary_dfs(
    session: base._Session,
    public: bool = True,
    ambiguous_tisues: List[str] = ["COLON", "BLOOD", "BONE_MARROW", "BRAIN"],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Builds summary study, contrast, and sample tables.

    Args:
        session (base._Session): SQLAlchemy session object to the main postgres/sqlite db
        public (bool): Filter for studies without public flag set
        ambiguous_tissues (List[str]): List of tissues (sample_type_1) where we want to
            transfer higher-granularity labels

    Returns:
        result (Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]):
            Tuple of (studies_df, contrasts_df, samples_df)
    """
    # Pull studies df.
    studies_df = fetch_studies(session, public=public)

    # Pull and munge metadata samples df.
    samples_df = fetch_samples(
        session,
        keep_fields=lambda x: x.startswith(
            (
                "sample_type_1",
                "sample_type_2",
                "sample_condition_1",
                "sample_condition_2",
                "disease_",
            )
        ),
        public=public,
    )
    samples_df["primary_disease"] = samples_df.apply(
        lambda x: [
            c
            for c in x[x.index.str.startswith("disease_")]
            if not pd.isna(c) and c != "NA"
        ],
        axis=1,
    )
    samples_df["primary_tissue"] = samples_df.apply(
        lambda x: (
            x["sample_type_2"].upper()
            if not pd.isna(x["sample_type_2"])
            and x["sample_type_1"] in ambiguous_tisues
            else x["sample_type_1"].upper()
        ),
        axis=1,
    )
    samples_df["primary_condition"] = samples_df["sample_condition_1"].copy()

    # Pull samplecontrasts link table.
    contrasts_df = fetch_samplecontrasts(session, public=public)

    # Merge metadata into studies, contrasts.
    studies_df = (
        studies_df.loc[:, ["internal_id", "pmid", "title", "description", "created_at"]]
        .rename(
            {"created_at": "processed_at"},
            axis=1,
        )
        .merge(
            samples_df.loc[:, ["internal_id", "primary_tissue", "primary_disease"]],
            on="internal_id",
            how="left",
        )
        .explode("primary_disease")
        .drop_duplicates()
        .reset_index(drop=True)
    )

    contrasts_df = (
        contrasts_df.loc[
            contrasts_df["contrast_side"] == "left",
            ["internal_id", "contrast_name", "srx_id", "created_at"],
        ]
        .rename({"created_at": "processed_at"}, axis=1)
        .merge(
            samples_df.loc[
                :, ["internal_id", "primary_tissue", "primary_disease", "srx_id"]
            ],
            on=["internal_id", "srx_id"],
            how="left",
        )
        .loc[
            :,
            [
                "internal_id",
                "contrast_name",
                "primary_tissue",
                "primary_disease",
                "processed_at",
            ],
        ]
        .explode("primary_disease")
        .drop_duplicates()
        .reset_index(drop=True)
    )

    samples_df = samples_df.loc[
        :,
        [
            "srx_id",
            "internal_id",
            "primary_tissue",
            "primary_disease",
            "primary_condition",
            "created_at",
        ],
    ].rename(
        {"created_at": "processed_at"},
        axis=1,
    )

    return studies_df, contrasts_df, samples_df
