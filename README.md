We're **building the clustering/linking pipeline yourselves first**, locally, then we're going to give this to an agent with a stable command/process to run.

The best architecture is **layered pipeline**:

```txt
Drive files
→ leaf JSONs
→ embeddings
→ similarity graph
→ deterministic rules
→ clusters
→ hierarchy
→ manifest
→ wiki
```

## Techniques that help

### 1. Classification

This answers:

```txt
What kind of thing is this?
```

Use existing leaf fields first:

```js
classification.document_type
classification.business_area
classification.audience
entities.*
source.name
embedding.keywords
```

This gives departments/categories like:

```txt
security
sales
product
legal
customers
competitors
erps
partners
```

This should be mostly deterministic/rule-based at first.

### 2. Embedding similarity

This answers:

```txt
Which files talk about similar things?
```

Embed:

```js
leaf.embedding.text
```

Then compute cosine similarity between files.

This is good for discovering hidden connections like:

```txt
Pricing Calculator ↔ Pricing Proposal Deck ↔ Upsell Feedback
Sage ICP ↔ Sage Demo ↔ Sage GTM Plan
Security Policy ↔ Access Control ↔ Network Security
```

### 3. Graph building

This answers:

```txt
How are files connected?
```

Create edges like:

```json
{
  "from": "file-a",
  "to": "file-b",
  "type": "similarity",
  "score": 0.84,
  "reason": "Both discuss Sage Intacct ICP and GTM"
}
```

Also add rule-based edges:

```txt
same customer
same product
same ERP
same source topic
same business area
same named entity
```

This is more useful than plain clustering because one file can belong to multiple contexts.

### 4. Clustering

This answers:

```txt
What groups naturally exist?
```

Best options:

```txt
Hierarchical clustering
→ good for wiki/tree structure

HDBSCAN
→ good for discovering natural clusters without knowing count

K-means
→ only useful if you already know the number of clusters

KNN graph
→ good for “related pages” and similarity links
```

For your use case, I’d start with:

```txt
KNN graph + hierarchical clustering
```

Not k-means.

K-means forces every file into one cluster, but OUR wiki needs overlap. Example: “Pricing Models & ROI” belongs to Sales, Product, Finance, and Payments.

## Recommended pipeline

```txt
Step 1: Validate leaves
- required fields exist
- embedding.text exists
- source.drive_file_id exists
- promotion.ready_for_clustering is true

Step 2: Embed leaves
- create vector per leaf
- store vector locally or in a vector DB

Step 3: Build similarity graph
- for each leaf, find top 5–10 nearest neighbors
- create similarity edges

Step 4: Add deterministic edges
- shared entities
- shared products
- shared customers
- shared ERPs
- shared business_area
- shared audience

Step 5: Create clusters
- use graph communities or hierarchical clustering (I prefer hierarchical)
- produce cluster JSONs

Step 6: Human-readable hierarchy
- map clusters into wiki sections
- departments first
- references/entities second

I want these sections to be defined:
```txt
const SECTION_ORDER = [
  'topics/product',
  'topics/integrations',
  'topics/sales',
  'topics/marketing',
  'topics/channel',
  'topics/operations',
  'topics/hr',
  'topics/legal',
  'topics/compliance',
  'topics/payments',
  'topics/reconciliation',

  'entities/customers',
  'entities/competitors',
  'entities/erps',
  'entities/products',
  'entities/features',
  'entities/partners',
  'entities/people',

  'decisions',
  'meta'
];

const SECTION_LABELS = {
  'topics/product': 'Product',
  'topics/integrations': 'Engineering / Integrations',
  'topics/sales': 'Sales',
  'topics/marketing': 'Marketing',
  'topics/channel': 'Channel / Partners',
  'topics/operations': 'Finance / Operations',
  'topics/hr': 'HR / People',
  'topics/legal': 'Legal',
  'topics/compliance': 'Risk & Compliance',
  'topics/payments': 'Payments',
  'topics/reconciliation': 'Reconciliation',

  'entities/customers': 'Customers',
  'entities/competitors': 'Competitors',
  'entities/erps': 'ERPs',
  'entities/products': 'Products',
  'entities/features': 'Features',
  'entities/partners': 'Partners',
  'entities/people': 'People',

  'decisions': 'Decisions',
  'meta': 'Meta'
};
```


Step 7: Generate manifest
- pages
- source_files
- related_pages
- cluster membership

## Desired scalable output

I’d separate the output into three files:

```txt
leaves/
  file-<drive_file_id>.json

clusters/
  cluster-<slug>.json

manifest.json
```

Cluster example:

```json
{
  "cluster_id": "cluster-sales-pricing-roi",
  "title": "Sales / Pricing & ROI",
  "section": "topics/sales",
  "summary": "Files related to pricing, ROI, module costs, and deal economics.",
  "leaf_ids": [
    "file-abc",
    "file-def"
  ],
  "related_clusters": [
    "cluster-products-modules",
    "cluster-payments-incentives"
  ],
  "confidence": "high"
}
```

Manifest page example:

```json
"topic-sales-pricing-models-roi": {
  "drive_file_id": "generated-html-id",
  "section": "topics/sales",
  "title": "Pricing Models & ROI",
  "summary": "...",
  "source_files": [
    {
      "name": "Pricing Calculator",
      "drive_file_id": "...",
      "url": "..."
    }
  ],
  "leaf_ids": ["file-abc", "file-def"],
  "cluster_ids": ["cluster-sales-pricing-roi"],
  "related_pages": ["topic-payments-behavioral-incentives"]
}
```

## Where to build it

Build it locally first, as a Python pipeline:

```txt
wiki-pipeline/
  input/leaves/
  output/clusters/
  output/manifest.json
  scripts/
    validate_leaves
    embed_leaves
    build_graph
    build_clusters
    generate_manifest
```

Then the agent only runs:

```bash
npm run wiki:cluster
```

or:

```bash
python pipeline.py
```

The agent should not decide structure freely. It should follow your pipeline and output schema.

## My recommendation

Start simple:

```txt
1. Deterministic grouping by business_area / section
2. Add embedding top-neighbor links
3. Generate related_pages
4. Later add hierarchical clustering
```

The first useful version is not “perfect AI clustering.” It is:

```txt
Every file has neighbors.
Every file has a source.
Every page knows its source files.
Every cluster is reproducible.
```

That is the scalable foundation.
