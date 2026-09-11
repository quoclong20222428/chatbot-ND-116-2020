\set ON_ERROR_STOP on

BEGIN;

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
	document_id TEXT PRIMARY KEY,
	document_number TEXT,
	document_title TEXT NOT NULL,
	source_type TEXT NOT NULL CHECK (source_type IN ('core', 'reference', 'qa')),
	document_role TEXT NOT NULL CHECK (document_role IN ('primary', 'amendment', 'supporting', 'qa')),
	authority_level TEXT NOT NULL CHECK (authority_level IN ('primary_legal_source', 'amending_legal_source', 'supporting_legal_source', 'reference_qa')),
	retrieval_priority INTEGER NOT NULL CHECK (retrieval_priority BETWEEN 0 AND 100),
	retrieval_behavior TEXT NOT NULL,
	created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS legal_chunks (
	chunk_id TEXT PRIMARY KEY,
	document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
	document_title TEXT NOT NULL,
	document_number TEXT,
	source_type TEXT NOT NULL CHECK (source_type IN ('core', 'reference', 'qa')),
	document_role TEXT NOT NULL CHECK (document_role IN ('primary', 'amendment', 'supporting', 'qa')),
	authority_level TEXT NOT NULL CHECK (authority_level IN ('primary_legal_source', 'amending_legal_source', 'supporting_legal_source', 'reference_qa')),
	retrieval_priority INTEGER NOT NULL CHECK (retrieval_priority BETWEEN 0 AND 100),
	retrieval_behavior TEXT NOT NULL,
	document_relations JSONB NOT NULL DEFAULT '[]'::JSONB,
	chapter TEXT,
	section TEXT,
	subsection TEXT,
	article TEXT,
	clause TEXT,
	point TEXT,
	form_number TEXT,
	content_type TEXT NOT NULL DEFAULT 'legal_text',
	question TEXT,
	answer TEXT,
	related_provisions JSONB NOT NULL DEFAULT '[]'::JSONB,
	text TEXT NOT NULL CHECK (btrim(text) <> ''),
	"references" JSONB NOT NULL DEFAULT '[]'::JSONB,
	search_vector TSVECTOR GENERATED ALWAYS AS (
		to_tsvector('simple', coalesce(document_title, '') || ' ' || coalesce(text, ''))
	) STORED,
	created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
	CHECK (jsonb_typeof("references") = 'array'),
	CHECK (jsonb_typeof(related_provisions) = 'array'),
	CHECK (jsonb_typeof(document_relations) = 'array')
);

CREATE TABLE IF NOT EXISTS legal_chunk_references (
	reference_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
	chunk_id TEXT NOT NULL REFERENCES legal_chunks(chunk_id) ON DELETE CASCADE,
	raw_reference TEXT NOT NULL,
	resolved_document_id TEXT REFERENCES documents(document_id) ON DELETE RESTRICT,
	resolved_chunk_id TEXT REFERENCES legal_chunks(chunk_id) ON DELETE RESTRICT,
	provision TEXT,
	UNIQUE (chunk_id, raw_reference, resolved_document_id, resolved_chunk_id, provision)
);

CREATE INDEX IF NOT EXISTS legal_chunks_document_idx
	ON legal_chunks (document_id);
CREATE INDEX IF NOT EXISTS legal_chunks_source_type_idx
	ON legal_chunks (source_type);
CREATE INDEX IF NOT EXISTS legal_chunks_role_priority_idx
	ON legal_chunks (document_role, retrieval_priority DESC);
CREATE INDEX IF NOT EXISTS legal_chunks_hierarchy_idx
	ON legal_chunks (document_id, article, clause, point);
CREATE INDEX IF NOT EXISTS legal_chunks_content_type_idx
	ON legal_chunks (content_type);
CREATE INDEX IF NOT EXISTS legal_chunks_search_vector_idx
	ON legal_chunks USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS legal_chunks_text_trgm_idx
	ON legal_chunks USING GIN (text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS legal_chunk_references_chunk_idx
	ON legal_chunk_references (chunk_id);
CREATE INDEX IF NOT EXISTS legal_chunk_references_document_idx
	ON legal_chunk_references (resolved_document_id);

CREATE TEMP TABLE _legal_chunks_import (
	record JSONB NOT NULL
) ON COMMIT DROP;

-- Run this script from the repository root so the relative path resolves.
\copy _legal_chunks_import(record) FROM 'data/processed/legal_chunks.jsonl'

DELETE FROM legal_chunk_references;

DELETE FROM legal_chunks AS chunk
WHERE NOT EXISTS (
	SELECT 1
	FROM _legal_chunks_import AS incoming
	WHERE incoming.record->>'chunk_id' = chunk.chunk_id
);

INSERT INTO documents (
	document_id,
	document_number,
	document_title,
	source_type,
	document_role,
	authority_level,
	retrieval_priority,
	retrieval_behavior
)
SELECT DISTINCT ON (record->>'document_id')
	record->>'document_id',
	NULLIF(record->>'document_number', ''),
	record->>'document_title',
	record->>'source_type',
	record->>'document_role',
	record->>'authority_level',
	(record->>'retrieval_priority')::INTEGER,
	record->>'retrieval_behavior'
FROM _legal_chunks_import
WHERE record->>'document_id' IS NOT NULL
ORDER BY record->>'document_id'
ON CONFLICT (document_id) DO UPDATE SET
	document_number = EXCLUDED.document_number,
	document_title = EXCLUDED.document_title,
	source_type = EXCLUDED.source_type,
	document_role = EXCLUDED.document_role,
	authority_level = EXCLUDED.authority_level,
	retrieval_priority = EXCLUDED.retrieval_priority,
	retrieval_behavior = EXCLUDED.retrieval_behavior,
	updated_at = now();

INSERT INTO legal_chunks (
	chunk_id,
	document_id,
	document_title,
	document_number,
	source_type,
	document_role,
	authority_level,
	retrieval_priority,
	retrieval_behavior,
	document_relations,
	chapter,
	section,
	subsection,
	article,
	clause,
	point,
	form_number,
	content_type,
	question,
	answer,
	related_provisions,
	text,
	"references"
)
SELECT
	record->>'chunk_id',
	record->>'document_id',
	record->>'document_title',
	NULLIF(record->>'document_number', ''),
	record->>'source_type',
	record->>'document_role',
	record->>'authority_level',
	(record->>'retrieval_priority')::INTEGER,
	record->>'retrieval_behavior',
	COALESCE(record->'document_relations', '[]'::JSONB),
	NULLIF(record->>'chapter', ''),
	NULLIF(record->>'section', ''),
	NULLIF(record->>'subsection', ''),
	NULLIF(record->>'article', ''),
	NULLIF(record->>'clause', ''),
	NULLIF(record->>'point', ''),
	NULLIF(record->>'form_number', ''),
	COALESCE(NULLIF(record->>'content_type', ''), 'legal_text'),
	NULLIF(record->>'question', ''),
	NULLIF(record->>'answer', ''),
	COALESCE(record->'related_provisions', '[]'::JSONB),
	record->>'text',
	COALESCE(record->'references', '[]'::JSONB)
FROM _legal_chunks_import
ON CONFLICT (chunk_id) DO UPDATE SET
	document_id = EXCLUDED.document_id,
	document_title = EXCLUDED.document_title,
	document_number = EXCLUDED.document_number,
	source_type = EXCLUDED.source_type,
	document_role = EXCLUDED.document_role,
	authority_level = EXCLUDED.authority_level,
	retrieval_priority = EXCLUDED.retrieval_priority,
	retrieval_behavior = EXCLUDED.retrieval_behavior,
	document_relations = EXCLUDED.document_relations,
	chapter = EXCLUDED.chapter,
	section = EXCLUDED.section,
	subsection = EXCLUDED.subsection,
	article = EXCLUDED.article,
	clause = EXCLUDED.clause,
	point = EXCLUDED.point,
	form_number = EXCLUDED.form_number,
	content_type = EXCLUDED.content_type,
	question = EXCLUDED.question,
	answer = EXCLUDED.answer,
	related_provisions = EXCLUDED.related_provisions,
	text = EXCLUDED.text,
	"references" = EXCLUDED."references";

INSERT INTO legal_chunk_references (
	chunk_id,
	raw_reference,
	resolved_document_id,
	resolved_chunk_id,
	provision
)
SELECT
	chunk.chunk_id,
	reference->>'raw_reference',
	NULLIF(reference->>'resolved_document_id', ''),
	NULLIF(reference->>'resolved_chunk_id', ''),
	NULLIF(reference->>'provision', '')
FROM legal_chunks AS chunk
CROSS JOIN LATERAL jsonb_array_elements(chunk."references") AS reference
WHERE NULLIF(reference->>'raw_reference', '') IS NOT NULL
ON CONFLICT DO NOTHING;

COMMIT;
