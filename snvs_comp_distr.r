.libPaths(c('~/Rlibs',.libPaths()))
suppressWarnings(library('optparse'))

script.desc <-
'Generates statistics to determine if SNV distribution has role in clustering.
User must supply the RDS of a Seurat object (-r) which should contain a single
Seurat experiment. A file also must be submitted with (-t) which is a file that
contains a path to all the scReadCounts files containing the SNV information.'

parser <- OptionParser(description=script.desc)
parser <- add_option(parser, c('-r', '--rds-file'),
                     type='character',
                     help='RDS file containing Seurat object.')
parser <- add_option(parser, c('-t', '--snv-file'),
                     type='character',
                     help='scReadCounts output file containing SNVs.')
parser <- add_option(parser, c('-w', '--th-snv-cells'),
                     type='numeric', default=10,
                     help=paste0('Threshold for maximum percentage of cells ',
				 ' that contain an SNV for bad reads. ', 
				 '(Default: 10)'))
parser <- add_option(parser, c('-z', '--th-snv-reads'),
                     type='numeric', default=1,
           help=paste0('Threshold for number of minimum reads to ',
                       'qualify as SNV. (Default: 1)'))
args <- parse_args(parser)

error.msg <- NULL
# Check if Seurat object is passed
if (is.null(args$`rds-file`))
  error.msg <- paste(error.msg, '- Seurat RDS object (-r) is required.',
                     sep='\n')

# Check if SNV argument is passed
if (is.null(args$`snv-file`))
  error.msg <- paste(error.msg, '- SNV data (-t) is required.', sep='\n')

if (args$`th-snv-reads`<=0)
  error.msg <- paste(error.msg, '- th-snv-reads needs to be greater than 0.',
                     sep='\n')

# Check if there are any errors before proceeding
if (!is.null(error.msg)) {
  print_help(parser)
  stop(error.msg)
}

library('Seurat')
srat <- readRDS(args$`rds-file`)
snv <- read.table(args$`snv-file`, sep= '\t', header=T)
# convert '-' to NA in VAF column (here VAF = ∞ or not defined as var, ref = 0)
snv$VAF[snv$VAF == '-'] <- NA
# convert VAF to numeric
snv$VAF <- as.numeric(snv$VAF)

sample.name <- gsub('(.*/)*([A-Za-z0-9]+)_.*.rds', '\\2', args$`rds-file`)

# Filtering based on arguments passed
# First filter (-x)
#snv <- snv[snv$X.BadRead<=args$`th-bad-reads`, ]

# First filter (-w)
snv.temp <- snv
snv.temp['BadReadFlag'] = 0
snv.temp[snv.temp[['X.BadRead']]>0, 'BadReadFlag'] = 1
snv.read.filt <- aggregate(BadReadFlag~CHROM+POS+REF+ALT, data=snv.temp,
                           function (x) 100*sum(x)/length(x))
snv.read.filt <- snv.read.filt[snv.read.filt$BadReadFlag<=args$`th-snv-cells`, 
                               c('CHROM', 'POS', 'REF', 'ALT')]
snv <- merge(snv, snv.read.filt, by=c('CHROM', 'POS', 'REF', 'ALT'))

# Second filter (-z)
snv$VAF[snv$SNVCount<args$`th-snv-reads`] <- 0

if (nrow(snv)==0)
  stop('There are no rows left after filtering.')


# Cluster
df.cid <- as.data.frame(srat[['seurat_clusters']])
df.cid <- data.frame(ReadGroup=rownames(df.cid),
                     ClusterID=df.cid[, 1], row.names=NULL)
n.cluster <- length(unique(df.cid$ClusterID))
df.snv <- merge(snv, df.cid, by='ReadGroup')

# Calculate variance of VAF across the sample
df.snv$VAF[df.snv$VAF>0] <- 1 # Change non-zero VAF to 1
df.samp.stats <- aggregate(VAF~CHROM+POS+REF+ALT, data=df.snv,
                           function (x) sum((x-mean(x, na.rm=T))^2, na.rm=T))
colnames(df.samp.stats)[5] <- 'TotalSS'


# Calculate estimate VAF for each cluster based on mean in a cluster
df.est <- aggregate(VAF~CHROM+POS+REF+ALT+ClusterID, data=df.snv,
                    function (x) mean(x, na.rm=T))
colnames(df.est)[6] <- 'ModelVAF'


# Calculate number of cells for each SNV
df.n.int <- aggregate(VAF~CHROM+POS+REF+ALT, data=df.snv,
                      function (x) sum(!is.na(x)))
colnames(df.n.int)[5] <- 'N'

# Calculate model prediction
df.clust.int <- merge(df.snv, df.est, by=c('CHROM', 'POS', 'REF', 'ALT',
                      'ClusterID'))
df.clust.int <- merge(df.clust.int, df.n.int, by=c('CHROM', 'POS', 'REF',
                      'ALT'))
df.clust.int <- df.clust.int[, c(1, 2, 3, 4, 5, 6, 15, 16, 17)]
df.clust.int['ModelSS'] <- (df.clust.int['VAF']-df.clust.int['ModelVAF'])^2

df.clust.int['ModelSS'] <- df.clust.int['ModelSS']
df.clust.sum <- aggregate(ModelSS~CHROM+POS+REF+ALT, data=df.clust.int,
                          function (x) sum(x, na.rm=T))

df.final <- merge(df.samp.stats, df.clust.sum,
                  by=c('CHROM', 'POS', 'REF', 'ALT'))
df.final <- merge(df.final, df.n.int,
                  by=c('CHROM', 'POS', 'REF', 'ALT'))

# Do basic filtering
df.final['R2'] <- 1-df.final['ModelSS']/df.final['TotalSS']
df.final['F'] <- ((df.final['TotalSS']-df.final['ModelSS'])/(n.cluster-1))/
                  (df.final['ModelSS']/(df.final['N']-n.cluster))

# Do basic filtering
df.final <- df.final[df.final$TotalSS!=0, ]
df.final <- df.final[is.finite(df.final$F), ]
df.final <- df.final[df.final$F>0, ]

# Calculate p-value
df.final['p'] <- 1-pf(df.final$F, n.cluster-1,df.final$N-n.cluster)
df.final['padj'] <- p.adjust(df.final$p, method='bonferroni',
                    length(df.final$p))
df.final <- df.final[order(df.final$padj, -df.final$F), ]

write.table(df.final, paste0(sample.name, '_snv_candidate_',
                             sprintf('%02d', args$`th-snv-cells`), 'n',
                             sprintf('%02d', args$`th-snv-reads`),
                             'n.txt'),
            sep='\t', row.names=F)
df.final <- df.final[df.final$padj < 0.1, ]
write.table(df.final, paste0(sample.name, '_snv_candidate_',
                             sprintf('%02d', args$`th-snv-cells`), 'n',
                             sprintf('%02d', args$`th-snv-reads`),
                             'n_q01.txt'),
            sep='\t', row.names=F)
