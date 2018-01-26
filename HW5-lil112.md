HW5-lil112
================
Liping Li (<lil112@pitt.edu>)
2017.03.28

Task 1: aanalyze the topical clusters from text data
====================================================

### Load data

``` r
data.file = "http://www.yurulin.com/class/spring2017_datamining/data/Newsgroup.csv"
NewsGroup = read.csv(data.file) 
NewsGroup[1:3,]
```

    ##         Topic
    ## 1 alt.atheism
    ## 2 alt.atheism
    ## 3 alt.atheism
    ##                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              Content
    ## 1 amus atheist and agnost articl timmbak mcl timmbak mcl ucsb edu clam bake timmon write fallaci atheism faith hear faq beckon onc wonder rule delet you correct you didn conspiraci correct hard atheism faith rule don mix appl orang can you that extermin mongol wors stalin khan conquer peopl unsympathet that atroci stalin kill million peopl love and worship and atheist state can wors that will not explain thi you stalin noth name atheism wheth not atheist irrelev grip man stalin exampl brought not indict atheism anoth exampl peopl will kill name that fit for occas you never implic pretti clear can respond your word not your true mean usenet slipperi medium delet wrt burden proof hard atheism noth prove justifi that god not exist faq etc guess justif compel aren peopl flock hard atheism not and won for will discourag peopl hard atheism point sourc reliabl statement hard atheism not support dogmat posit fool that larg group peopl that atheist peopl exist proselyt fashion religion hard atheist you post not hard second make you defend religion recogn hard atheism for faith never meant understand you that idea bibl exampl allegori illustr point and refer reader post evid that poster state that reli evid for lost thi thread theist arrog delet such and such absolut unalter true dogma true not prepar issu blanket statement indict theist arrog you wont atheist bzzt virtu your innoc pronoun you issu blanket statement will apolog qualifi origin statement hard atheist place atheist you call john baptist arrog boast greater that christian todai that arrog guilti charg meant theist arrog thi that thought mean clear posit that claim superior anoth support evid arrog for your apolog btw not worthi misinform your sophist put theist your misinform shine explain bake timmon iii noth higher stronger wholesom and life good memori alyosha brother karamazov dostoevski
    ## 2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    rushdi islam law jaeger buphi edu gregg jaeger write articl vice ico tek bobb vice ico tek robert beauchain write bennett neil bcci adapt koran rule bank time august gui write piec titl that impli case that gregg you haven provid even titl articl support your content thi you support posit you intend respect gregg question and even manag includ refer head firmli engag ass your excus thi support noth reason that thi piec anoth anti islam slander job you reason anti islam slander job apart your prejudic respect for titl for real content can thi articl want true can you bcci not islam bank mere report time state that bcci islam bank rule gregg islam good and bcci bad bcci cannot islam spread slander propaganda discuss issu glad real discuss provid refer etc provid refer articl you agre you will respond refer articl you agre mmm that intellectu stimul debat doubtless that you spend your time soc cultur islam special place for you kill file bobbi want join you post becom convinc that simpli wast time and reason moslem that you hope achiev mathew
    ## 3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              christian moral articl vice ico tek bobb vice ico tek robert beauchain write not believ god not concern disposit beneath provid evid requir evid that person thi god find compel fact god you you for minut you love you you made love you want love you respons you love god and step promis for you you for doubt thi disput not givin sincer effort simpl logic argument folli you read bibl you will that jesu made fool trick logic abil reason spec creation ultim you reli simpli your reason you will never you learn you accept that you don

1.1 Popular Topics
------------------

Plot the histogram of number of documents per topic. Find and list the four most popular topics in terms of number of documents.

### Histogram of Topic frequency

``` r
library(ggplot2)
ggplot(NewsGroup, aes(x=Topic)) + geom_histogram(stat = "count")
```

<img src="HW5-lil112_files/figure-markdown_github/histogram topics-1.png" style="display: block; margin: auto auto auto 0;" />

Most topics have more than 250 documents. And 1 topic without name has less than 10 documents.

### 4 popular topics

``` r
toptable = as.data.frame(table(NewsGroup$Topic))
toptable = toptable[order(toptable$Freq, decreasing = T),]
toptable[1:4,]
```

    ##                  Var1 Freq
    ## 10    rec.motorcycles  398
    ## 11 rec.sport.baseball  397
    ## 12   rec.sport.hockey  397
    ## 9           rec.autos  395

Totally 21 topics. **rec.motorcycles, rec.sport.baseball, rec.sport.hockey, and rec.auto** are most popular 4 topics.

1.2 MDS of 4 popular content
----------------------------

Extract contents in these top 4 topics as your corpus. Run pre-processing on this corpus and use terms that appear at least four times in the corpus to create a term-document matrix. Use the term-document matrix to generate an MDS plot where each node represents a document with color indicating its topic.

### Extract documents of 4 popular topics

``` r
select.topics = as.vector(toptable[1:4,1])

library(plyr)
doc.idx = which(NewsGroup$Topic %in% select.topics)
dataset = NewsGroup[doc.idx, ]

library(tm)
```

    ## Loading required package: NLP

    ## 
    ## Attaching package: 'NLP'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     annotate

``` r
## create a corpus
corpus = Corpus(VectorSource(dataset$Content))
corpus
```

    ## <<SimpleCorpus>>
    ## Metadata:  corpus specific: 1, document level (indexed): 0
    ## Content:  documents: 1587

### Preprocess Corpus

``` r
corpus = tm_map(corpus, tolower)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, function(x) removeWords(x, stopwords("english")))
corpus = tm_map(corpus, stemDocument, language = "english")
corpus = tm_map(corpus, stripWhitespace)
inspect(corpus[1:3])
```

    ## <<SimpleCorpus>>
    ## Metadata:  corpus specific: 1, document level (indexed): 0
    ## Content:  documents: 3
    ## 
    ## [1] auto air condit freon articl hvx new cso uiuc edu tspila uxa cso uiuc edu tim spila romulan write articl apr ntuix ntu mgqlu ntuix ntu max write work ga solid adsorpt air con system auto applic thi kind system energi regen adsorb exhaust ga interest thi mail email follow thi thread discuss prospect thi technolog bite thi suppo work tim year ago demonstr cold air system us air call rovax unit work short come seal technolog todai
    ## [2] auto air condit freon rovax tobia convex allen tobia write year ago demonstr cold air system us air call rovax unit work short come seal technolog todai recal read post back rovax rovac di larger noisier compet cheap system dai case bad time system chanc todai system death row investor hard come second time jon hacker march beta rom caltech pasadena call ibm hacker tumbler ridg caltech edu read comp beta                        
    ## [3] auto air condit freon simpl principl porou adsorb zeolit activ carbon can adsorb gase evapor adsorb water methanol etc give cool effect heat ga satur adsorb bed will give gase conden thi form adsorpt refrig cycl problem cop low max max phd internet mgqlu ntu divi thermal engin bitnet mgqlu ntuvax bitnet school mpe nanyang technolog univ phone nanyang avenu singapor fax

``` r
#corpus = tm_map(corpus, PlainTextDocument)
td.mat = TermDocumentMatrix(corpus)
dim(td.mat)
```

    ## [1] 13547  1587

Rows are terms, and columns are documents.

### Term frequency &gt;= 4

``` r
term.idx=c(findFreqTerms(td.mat, lowfreq=4))
td.mat = td.mat[term.idx, ]
dim(td.mat)
```

    ## [1] 5313 1587

### MDS plot based on TermDocumentMatrix

``` r
mdsplot <- function(td.mat)
{
dist.mat = dist(t(as.matrix(td.mat)))  ## compute distance matrix

doc.mds = cmdscale(dist.mat, k = 2)
data = data.frame(x = doc.mds[, 1], y = doc.mds[, 2], topic = dataset$Topic, id = row.names(dataset))
library(ggplot2)
ggplot(data, aes(x = x, y = y, color = topic)) + geom_point()
}
mdsplot(td.mat)
```

<img src="HW5-lil112_files/figure-markdown_github/mds for 4 popular-1.png" style="display: block; margin: auto auto auto 0;" />

These topics do not have a good separation from each other. Maybe they're all from recreation or sports group.

1.3 MDS based on different matrix
---------------------------------

Apply TFIDF weighting, latent semantic analysis (LSA) and non-negative matrix factorization (NMF) on the term-document matrix. Generate MDS plots corresponding to these matrices (TFIDF weighted matrix, LSA approximated matrix, and NMF approximated matrix).

### MDS plot based on TFIDF

``` r
library(lsa)
```

    ## Loading required package: SnowballC

``` r
td.mat = as.matrix(td.mat)
td.mat.w <- lw_tf(td.mat) * gw_idf(td.mat)  ## tf-idf weighting
mdsplot(td.mat.w)
```

<img src="HW5-lil112_files/figure-markdown_github/mds TFIDF-1.png" style="display: block; margin: auto auto auto 0;" />

### MDS plot based on LSA

``` r
lsa.space = lsa(td.mat.w,dims=4)  ## create LSA space, 4 topics
lsa.mat = as.textmatrix(lsa.space)
mdsplot(lsa.mat)
```

<img src="HW5-lil112_files/figure-markdown_github/mds LSA-1.png" style="display: block; margin: auto auto auto 0;" />

### MDS plot based on NMF

``` r
suppressMessages(library(NMF))
set.seed(12345)
res = nmf(td.mat, 4,"lee") # lee & seung method, 4 topics
V.hat = fitted(res) # target matrix
mdsplot(V.hat)
```

<img src="HW5-lil112_files/figure-markdown_github/mds NMF-1.png" style="display: block; margin: auto auto auto 0;" />

Try with Topic x Document matrix.

``` r
h = coef(res)
h = as.matrix(h)
mdsplot(h)
```

![](HW5-lil112_files/figure-markdown_github/nmf%20basis-1.png)

1.4 Observation Summary
-----------------------

All these plots seem alike, as well as NMF Topic x Document matrix MDS plot. rec.autos is on the top and has the largest range on x-axis and smallest range on y-axis, then rec.motorcycles, rec.sport.baseball and re.sport.hockey follow successively with increasing range on y-axis and decreasing range on x-axis.

All these topic groups coverge at (0,0) and have a big overlap. Maybe because they are all associate with recreation or sports.

``` r
rm(list = ls())
```
