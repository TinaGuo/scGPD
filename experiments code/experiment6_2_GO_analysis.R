suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(GOSemSim)
})

count <- readRDS("/Users/yunshanguo/Downloads/lung_n_t.rds")
bg_raw <- rownames(count)[rowSums(count) != 0]

### detect ID type for background
bg_is_ens <- any(grepl("^ENSG", bg_raw))

### helper: map IDs to ENTREZ safely
map_to_entrez <- function(ids, fromType){
  ids <- unique(ids)
  ids <- ids[!is.na(ids) & ids != ""]
  if (fromType == "ENSEMBL") {
    ids <- sub("\\..*$", "", ids)   ### drop version suffix like ".1"
  }
  mp <- bitr(ids,
             fromType = fromType,
             toType   = "ENTREZID",
             OrgDb    = org.Hs.eg.db)
  unique(mp$ENTREZID)
}

runGO2 <- function(gene_symbols, bg_ids){
  ### genes are SYMBOL (your genes_df$gene)
  gene_entrez <- map_to_entrez(gene_symbols, fromType = "SYMBOL")
  
  ### background may be ENS or SYMBOL
  bg_from <- if (any(grepl("^ENSG", bg_ids))) "ENSEMBL" else "SYMBOL"
  bg_entrez <- map_to_entrez(bg_ids, fromType = bg_from)
  
  ### sanity checks to avoid phyper NaNs
  stopifnot(length(gene_entrez) > 5)
  stopifnot(length(bg_entrez) > length(gene_entrez))
  
  enrichGO(
    gene          = gene_entrez,
    universe      = bg_entrez,
    OrgDb         = org.Hs.eg.db,
    keyType       = "ENTREZID",
    ont           = "BP",
    pAdjustMethod = "BH",
    minGSSize     = 20,
    maxGSSize     = 200,
    pvalueCutoff  = 1,
    qvalueCutoff  = 1,
    readable      = TRUE
  )
}

### GO BP
ego_BP <- runGO2(genes_df$gene, bg_raw)

### semantic similarity
semdata_bp <- godata('org.Hs.eg.db', ont = "BP")
ego2 <- pairwise_termsim(ego_BP, method = "Wang", semData = semdata_bp)
