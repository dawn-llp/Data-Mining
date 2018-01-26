HW4-lil112
================
Liping Li (<lil112@pitt.edu>)
2017.03.06

Task 1: analyze the data unempstates.csv
========================================

### Load data

``` r
data.file = "http://www.yurulin.com/class/spring2017_datamining/data/unempstates.csv"
unempstates = read.csv(data.file, header=T, sep=",", stringsAsFactors=T) 
unempstates[1:3, ]
```

    ##    AL  AK   AZ  AR  CA  CO  CT  DE   FL  GA  HI  ID  IL  IN  IA  KS  KY
    ## 1 6.4 7.1 10.5 7.3 9.3 5.8 9.4 7.7 10.0 8.3 9.9 5.5 6.4 6.9 4.2 4.3 5.7
    ## 2 6.3 7.0 10.3 7.2 9.1 5.7 9.3 7.8  9.8 8.2 9.8 5.4 6.4 6.6 4.2 4.2 5.6
    ## 3 6.1 7.0 10.0 7.1 9.0 5.6 9.2 7.9  9.5 8.1 9.6 5.4 6.4 6.4 4.1 4.2 5.5
    ##    LA  ME  MD   MA   MI  MN  MS  MO  MT  NE  NV  NH   NJ  NM   NY  NC  ND
    ## 1 6.2 8.8 6.9 11.1 10.0 6.2 7.0 5.8 5.8 3.6 9.8 7.2 10.5 8.9 10.2 6.7 3.2
    ## 2 6.2 8.6 6.7 10.9  9.9 6.0 6.8 5.8 5.7 3.5 9.5 7.1 10.4 8.8 10.2 6.5 3.3
    ## 3 6.2 8.5 6.6 10.6  9.8 5.8 6.6 5.8 5.7 3.3 9.3 7.0 10.4 8.7 10.1 6.3 3.3
    ##    OH  OK   OR  PA  RI  SC  SD  TN  TX  UT  VT  VA  WA  WV  WI  WY
    ## 1 8.3 6.4 10.1 8.1 7.8 7.6 3.6 5.9 5.9 6.1 8.8 6.2 8.7 8.3 5.9 4.2
    ## 2 8.2 6.3  9.8 8.1 7.8 7.4 3.5 5.9 5.9 5.9 8.7 6.1 8.7 8.1 5.7 4.1
    ## 3 8.0 6.1  9.5 8.0 7.9 7.2 3.4 5.9 5.8 5.7 8.6 5.9 8.7 7.9 5.6 4.0

Monthly seasonally adjusted unemployment rates covering the period January 1976 through August 2010 for the 50 US states(n = 50).

### Transpose data

``` r
unempstates = t(unempstates)
dim(unempstates)
```

    ## [1]  50 416

### 1.1 PCA screeplot & loadings for first principal component

``` r
pca.unemp = prcomp(unempstates,scale=TRUE) 
screeplot(pca.unemp, main="Unemployment Rate Principal Components", type="line") 
```

![](HW4-lil112_files/figure-markdown_github/pca%20unempstates-1.png)

``` r
#mtext(side=1, "European Protein Principal Components",  line=1, font=2)
```

There is an obvious slope change at "4", suggesting no more than 4 factors should be retained.

``` r
pc.unemp = predict(pca.unemp)
label = rownames(unempstates)
plot(pc.unemp[,1], type="h", main="Loading of 1st Principal Component",xlab="States")
text(x=pc.unemp[,1], labels=label, cex=.7)
```

![](HW4-lil112_files/figure-markdown_github/1st%20factor%20loading%20unempstates-1.png)

### 1.2 First Two Principal Components Projection

``` r
plot(pc.unemp[,1:2], type="n", main="First Two Principal Components Projection")
text(x=pc.unemp[,1], y=pc.unemp[,2], labels=label,cex=.7)
```

![](HW4-lil112_files/figure-markdown_github/2%20factor%20loadings%20unempstates-1.png)

### 1.3 MDS Map of Unemployment Data from all states

``` r
unemp.dist = dist(unempstates)
unemp.mds = cmdscale(unemp.dist)
plot(unemp.mds, type = 'n',main="MDS Map of Unemployment Data")
text(unemp.mds, labels=label, cex=.7)
```

![](HW4-lil112_files/figure-markdown_github/2%20MDS%20map%20unempstates-1.png)

MDS map resembles PCA projection plot except the direction of y-axis.

### 1.4 Clustering in MDS maps and plot dendrograms

At first, I'm worried about scaling method before clustering. But all the columns are unemployment rate, I

``` r
set.seed(12312)
# K-means, k = 4
km4 = kmeans(unempstates, centers=4, nstart=10)

# K-means, k = 8
km8 = kmeans(unempstates, centers=8, nstart=10)

# use library 'cluster' for hierarchy clustering
library(cluster)
# h-clustering with single-link, k = 4, k = 8
single.hc = hclust(unemp.dist,method='single')
single.hc4 = cutree(single.hc,k=4)
single.hc8 = cutree(single.hc,k=8)

# h-clustering with complete-link, k = 4, k = 8
complete.hc = hclust(unemp.dist,method='complete')
complete.hc4 = cutree(complete.hc,k=4)
complete.hc8 = cutree(complete.hc,k=8)

# h-clustering with average-link , k = 4
average.hc = hclust(unemp.dist,method='average')
average.hc4 = cutree(average.hc,k=4)
average.hc8 = cutree(average.hc,k=8)
```

``` r
par(mfrow=c(3,1))
plot(single.hc, main="h-clustering with single-link")
plot(complete.hc, main="h-clustering with complete-link")
plot(average.hc, main="h-clustering with average-link")
```

![](HW4-lil112_files/figure-markdown_github/unemp%20hc%20dendrogram-1.png)

``` r
par(mfrow=c(4,2))
plot(unemp.mds, type = 'n',main="MDS Map with K-mean 4 Groups")
text(unemp.mds, labels=label, cex=.7, col=rainbow(4)[km4$cluster])
plot(unemp.mds, type = 'n',main="MDS Map with K-mean 8 Groups")
text(unemp.mds, labels=label, cex=.7, col=rainbow(8)[km8$cluster])
plot(unemp.mds, type = 'n',main="MDS Map with single HC 4 Groups")
text(unemp.mds, labels=label, cex=.7, col=rainbow(4)[single.hc4])
plot(unemp.mds, type = 'n',main="MDS Map with single HC 8 Groups")
text(unemp.mds, labels=label, cex=.7, col=rainbow(8)[single.hc8])
plot(unemp.mds, type = 'n',main="MDS Map with complete HC 4 Groups")
text(unemp.mds, labels=label, cex=.7, col=rainbow(4)[complete.hc4])
plot(unemp.mds, type = 'n',main="MDS Map with complete HC 8 Groups")
text(unemp.mds, labels=label, cex=.7, col=rainbow(8)[complete.hc8])
plot(unemp.mds, type = 'n',main="MDS Map with average HC 4 Groups")
text(unemp.mds, labels=label, cex=.7, col=rainbow(4)[average.hc4])
plot(unemp.mds, type = 'n',main="MDS Map with average HC 8 Groups")
text(unemp.mds, labels=label, cex=.7, col=rainbow(8)[average.hc8])
```

![](HW4-lil112_files/figure-markdown_github/unemp%20clustering%20mds-1.png)

### 1.5 Summary

Based on these information, I think **K-means, k=4** and **h-clustering with complete-link, k=4** are better. For k=8, each method produces **overlapping** areas. For k=4, single-link and average-link conditions, **1 dominate class** occupy too big space, which might yields poor classification value.

Task 2: analyze US Senator Roll Call Data
=========================================

### Load data

``` r
library('foreign') ## for loading dta files using read.dta

data.url = 'http://www.yurulin.com/class/spring2017_datamining/data/roll_call/sen113kh.dta'
senvote = read.dta(data.url,convert.factors = FALSE)
dim(senvote)
```

    ## [1] 106 666

``` r
senvote[1:3,1:15]
```

    ##   cong    id state dist  lstate party eh1 eh2        name V1 V2 V3 V4 V5
    ## 1  113 99911    99    0 USA       100  NA  NA OBAMA        9  9  9  1  1
    ## 2  113 49700    41    0 ALABAMA   200   0   1 SESSIONS     6  6  1  6  1
    ## 3  113 94659    41    0 ALABAMA   200   0   1 SHELBY       6  6  6  1  1
    ##   V6
    ## 1  9
    ## 2  6
    ## 3  6

The first nine columns of the data frame include identification information for those voters, and the remaining columns are the actual votes.

### 2.1 MDS Map of Senators

``` r
unique(senvote$party)
```

    ## [1] 100 200 328

Map party Number into party name. Save a variable for color identity.

``` r
no.pres <- subset(senvote, state < 99)
party = no.pres$party # president happens to be the first line
party.color = floor(party/100)
party = as.character(party)
party = replace(party, party=="100", "Dem")
party = replace(party, party=="200", "Rep")
party = replace(party, party=="328", "Ind")
```

``` r
# rewrite class07 sample code
rollcall.simplified <- function(df) {
  no.pres <- subset(df, state < 99) # state =99, president
  ## to group all Yea and Nay types together
  for(i in 10:ncol(no.pres)) {
    no.pres[,i] = ifelse(no.pres[,i] > 6, 0, no.pres[,i])
    no.pres[,i] = ifelse(no.pres[,i] > 0 & no.pres[,i] < 4, 1, no.pres[,i])
    no.pres[,i] = ifelse(no.pres[,i] > 1, -1, no.pres[,i])
  }

  return(as.matrix(no.pres[,10:ncol(no.pres)]))
}

rollcall.simple = rollcall.simplified(senvote)

## Multiply the matrix by its transpose to get Senator-to-Senator tranformation, 
## and calculate the Euclidan distance between each Senator.
rollcall.dist =dist(rollcall.simple %*% t(rollcall.simple))

## Do the MDS
rollcall.mds =cmdscale(rollcall.dist, k = 2) * -1
# *-1 Make Dem on the left
```

\*\* Senator x Senator \*\*

``` r
plot(rollcall.mds, type = 'n',main="MDS Map of Senators x Senators")
text(rollcall.mds, labels=party, cex=.7,col=rainbow(3)[party.color])
```

![](HW4-lil112_files/figure-markdown_github/2%20MDS%20map%20Senators-1.png)

\*\* Senator- Voting MDS \*\*

``` r
rollcall.dist2 =dist(rollcall.simple)

rollcall.mds2 =cmdscale(rollcall.dist2, k = 2) * -1

plot(rollcall.mds2, type = 'n',main="MDS Map of Senators x Voting")
text(rollcall.mds2, labels=party, cex=.7,col=rainbow(3)[party.color])
```

![](HW4-lil112_files/figure-markdown_github/2%20MDS%20map%20Senators%20x%20Voting-1.png)

dist(rollcall.simple %\*% t(rollcall.simple)) returns greater scale, but has similar cmdscale patten with dist(rollcall.simple).

### 2.2 Clustering senators on MDS maps

``` r
set.seed(12312)
# K-means, k = 2
km2 = kmeans(no.pres[,-c(1:9)], centers=2, nstart=10)

# h-clustering with single-link, k = 2
single2 = hclust(rollcall.dist2,method='single')
single.hc2 = cutree(single2,k=2)

# h-clustering with complete-link, k = 2
complete2 = hclust(rollcall.dist2,method='complete')
complete.hc2 = cutree(complete2,k=2)

# h-clustering with average-link , k = 2
average2 = hclust(rollcall.dist2,method='average')
average.hc2 = cutree(average2, k=2)
```

``` r
par(mfrow=c(3,1))
plot(single2, main="h-clustering with single-link")
plot(complete2, main="h-clustering with complete-link")
plot(average2, main="h-clustering with average-link")
```

![](HW4-lil112_files/figure-markdown_github/vote%20hc%20dendrogram-1.png)

``` r
library(ggplot2) # complicate plot
library(gridExtra) # arrange plots 
plotdf = data.frame(rollcall.mds2,party,km2$cluster,single.hc2,complete.hc2,average.hc2) 
head(plotdf)
```

    ##           X1         X2 party km2.cluster single.hc2 complete.hc2
    ## 2  23.359483 -3.3390877   Rep           2          1            1
    ## 3  22.681085 -2.6550296   Rep           2          1            1
    ## 4   2.479482 10.3463284   Rep           2          2            2
    ## 5 -14.824995  0.2164311   Dem           1          1            2
    ## 6  16.741183  6.8809659   Rep           2          1            1
    ## 7  17.031809  6.6270164   Rep           2          1            1
    ##   average.hc2
    ## 2           1
    ## 3           1
    ## 4           2
    ## 5           2
    ## 6           1
    ## 7           1

``` r
kmplot = ggplot(plotdf, aes(x=X1, y=X2))+ geom_point(aes(shape=party, color=factor(km2.cluster)))
kmplot = kmplot + ggtitle("K-means clustering")
s2plot = ggplot(plotdf, aes(x=X1, y=X2))+ geom_point(aes(shape=party, color=factor(single.hc2)))
s2plot = s2plot + ggtitle("Hierarchical clustering, single-link")
c2plot = ggplot(plotdf, aes(x=X1, y=X2))+ geom_point(aes(shape=party, color=factor(complete.hc2)))
c2plot = c2plot + ggtitle("Hierarchical clustering, complete-link")
a2plot = ggplot(plotdf, aes(x=X1, y=X2))+ geom_point(aes(shape=party, color=factor(average.hc2)))
a2plot = a2plot + ggtitle("Hierarchical clustering, average-link")
grid.arrange(kmplot, s2plot, c2plot, a2plot, ncol=2)
```

![](HW4-lil112_files/figure-markdown_github/vote%20clustering%20plot%20block-1.png)

### 2.3 Identify wrongly labeled senators

**All independent senators are labeled with Democrats.** I don't further exploration on independent senators. Here only consider Democrats labeled as Republican and vice versa.

``` r
# sort clustering labels
km.result = plotdf$km2.cluster
km.result = replace(km.result, km.result==1, "Dem")
km.result = replace(km.result, km.result==2, "Rep")

hcs.result = replace(single.hc2, single.hc2==1, "Dem")
hcs.result = replace(hcs.result, hcs.result==2, "Rep")

hcc.result = replace(complete.hc2, complete.hc2==2, "Dem")
hcc.result = replace(hcc.result, hcc.result==1, "Rep")

hca.result = replace(average.hc2, average.hc2==2, "Dem")
hca.result = replace(hca.result, hca.result==1, "Rep")
```

``` r
groupCheck <- function(cluster,y){
  x1=0
  x2=0
  for (i in 1:105)
  {
    if(y[i]=="Dem" && y[i] != cluster[i])
      {
        cat("No.",i,"senator", no.pres$name[i],"from ",no.pres$lstate[i],"is a Democrats but labeled as Republican","\n")
        x1=x1+1
        }

    else if(y[i]=="Rep"&& y[i] != cluster[i] )
      {cat("No.",i,"senator", no.pres$name[i],"from ",no.pres$lstate[i],"is a Republican but labeled as Democrat","\n")
        x2=x2+1
      }
  }
  
  cat(x1, "Democrats are wrongly labeled as Republicans","\n")
  cat(x2, "Republicans are wrongly labeled as Democrates","\n")
}

print("K-means result comparison")
```

    ## [1] "K-means result comparison"

``` r
groupCheck(km.result,party)
```

    ## No. 37 senator COLLINS     from  MAINE   is a Republican but labeled as Democrat 
    ## No. 64 senator CHIESA      from  NEW JER is a Republican but labeled as Democrat 
    ## 0 Democrats are wrongly labeled as Republicans 
    ## 2 Republicans are wrongly labeled as Democrates

``` r
print("Single-link hierarchical clustering result comparison")
```

    ## [1] "Single-link hierarchical clustering result comparison"

``` r
groupCheck(hcs.result,party)
```

    ## No. 1 senator SESSIONS    from  ALABAMA is a Republican but labeled as Democrat 
    ## No. 2 senator SHELBY      from  ALABAMA is a Republican but labeled as Democrat 
    ## No. 5 senator FLAKE       from  ARIZONA is a Republican but labeled as Democrat 
    ## No. 6 senator MCCAIN      from  ARIZONA is a Republican but labeled as Democrat 
    ## No. 8 senator BOOZMAN     from  ARKANSA is a Republican but labeled as Democrat 
    ## No. 17 senator RUBIO       from  FLORIDA is a Republican but labeled as Democrat 
    ## No. 19 senator CHAMBLISS   from  GEORGIA is a Republican but labeled as Democrat 
    ## No. 20 senator ISAKSON     from  GEORGIA is a Republican but labeled as Democrat 
    ## No. 23 senator RISCH       from  IDAHO   is a Republican but labeled as Democrat 
    ## No. 24 senator CRAPO       from  IDAHO   is a Republican but labeled as Democrat 
    ## No. 26 senator KIRK        from  ILLINOI is a Republican but labeled as Democrat 
    ## No. 27 senator COATS       from  INDIANA is a Republican but labeled as Democrat 
    ## No. 29 senator GRASSLEY    from  IOWA    is a Republican but labeled as Democrat 
    ## No. 31 senator MORAN       from  KANSAS  is a Republican but labeled as Democrat 
    ## No. 32 senator ROBERTS     from  KANSAS  is a Republican but labeled as Democrat 
    ## No. 33 senator PAUL        from  KENTUCK is a Republican but labeled as Democrat 
    ## No. 34 senator MCCONNELL   from  KENTUCK is a Republican but labeled as Democrat 
    ## No. 35 senator VITTER      from  LOUISIA is a Republican but labeled as Democrat 
    ## No. 49 senator COCHRAN     from  MISSISS is a Republican but labeled as Democrat 
    ## No. 50 senator WICKER      from  MISSISS is a Republican but labeled as Democrat 
    ## No. 52 senator BLUNT       from  MISSOUR is a Republican but labeled as Democrat 
    ## No. 56 senator JOHANNS     from  NEBRASK is a Republican but labeled as Democrat 
    ## No. 57 senator FISCHER     from  NEBRASK is a Republican but labeled as Democrat 
    ## No. 58 senator HELLER      from  NEVADA  is a Republican but labeled as Democrat 
    ## No. 60 senator AYOTTE      from  NEW HAM is a Republican but labeled as Democrat 
    ## No. 64 senator CHIESA      from  NEW JER is a Republican but labeled as Democrat 
    ## No. 70 senator BURR        from  NORTH C is a Republican but labeled as Democrat 
    ## No. 73 senator HOEVEN      from  NORTH D is a Republican but labeled as Democrat 
    ## No. 75 senator PORTMAN     from  OHIO    is a Republican but labeled as Democrat 
    ## No. 76 senator INHOFE      from  OKLAHOM is a Republican but labeled as Democrat 
    ## No. 77 senator COBURN      from  OKLAHOM is a Republican but labeled as Democrat 
    ## No. 81 senator TOOMEY      from  PENNSYL is a Republican but labeled as Democrat 
    ## No. 84 senator SCOTT       from  SOUTH C is a Republican but labeled as Democrat 
    ## No. 85 senator GRAHAM      from  SOUTH C is a Republican but labeled as Democrat 
    ## No. 86 senator THUNE       from  SOUTH D is a Republican but labeled as Democrat 
    ## No. 88 senator CORKER      from  TENNESS is a Republican but labeled as Democrat 
    ## No. 89 senator ALEXANDER   from  TENNESS is a Republican but labeled as Democrat 
    ## No. 90 senator CORNYN      from  TEXAS   is a Republican but labeled as Democrat 
    ## No. 91 senator CRUZ        from  TEXAS   is a Republican but labeled as Democrat 
    ## No. 92 senator LEE         from  UTAH    is a Republican but labeled as Democrat 
    ## No. 93 senator HATCH       from  UTAH    is a Republican but labeled as Democrat 
    ## No. 102 senator JOHNSON     from  WISCONS is a Republican but labeled as Democrat 
    ## No. 104 senator ENZI        from  WYOMING is a Republican but labeled as Democrat 
    ## No. 105 senator BARASSO     from  WYOMING is a Republican but labeled as Democrat 
    ## 0 Democrats are wrongly labeled as Republicans 
    ## 44 Republicans are wrongly labeled as Democrates

``` r
print("Complete-link hierarchical clustering result comparison")
```

    ## [1] "Complete-link hierarchical clustering result comparison"

``` r
groupCheck(hcc.result,party)
```

    ## No. 3 senator MURKOWSKI   from  ALASKA  is a Republican but labeled as Democrat 
    ## No. 37 senator COLLINS     from  MAINE   is a Republican but labeled as Democrat 
    ## No. 64 senator CHIESA      from  NEW JER is a Republican but labeled as Democrat 
    ## 0 Democrats are wrongly labeled as Republicans 
    ## 3 Republicans are wrongly labeled as Democrates

``` r
print("Average-link hierarchical clustering result comparison")
```

    ## [1] "Average-link hierarchical clustering result comparison"

``` r
groupCheck(hca.result,party)
```

    ## No. 3 senator MURKOWSKI   from  ALASKA  is a Republican but labeled as Democrat 
    ## No. 37 senator COLLINS     from  MAINE   is a Republican but labeled as Democrat 
    ## No. 64 senator CHIESA      from  NEW JER is a Republican but labeled as Democrat 
    ## 0 Democrats are wrongly labeled as Republicans 
    ## 3 Republicans are wrongly labeled as Democrates

K-means yields the best result, then complete-link and average-link follows. Single-link hierarchical clustering performs the poorest. No Democrats are wrongly labeled. Republicans have the risk to be wrongly assigned.

### 2.4 Purity and Entropy

``` r
## copy from class06.html
cluster.measure <- function(clusters, classes)
{cluster.purity <- sum(apply(table(classes, clusters), 2, max)) / length(clusters)

  en <- function(x) {
    s = sum(x)
    sum(sapply(x/s, function(p) {if (p) -p*log2(p) else 0} ) )
  }
  M = table(classes, clusters)
  m = apply(M, 2, en)
  c = colSums(M) / sum(M)
  cluster.entropy = sum(m*c)
  
  return(c(cluster.purity,cluster.entropy))
}
```

``` r
km.measure = cluster.measure(km.result,party)
hcs.measure = cluster.measure(hcs.result,party)
hcc.measure = cluster.measure(hcc.result,party)
hca.measure = cluster.measure(hca.result,party)
measureTab = cbind(km.measure,hcs.measure,hcc.measure,hca.measure)
row.names(measureTab) =c("Purity","Entropy")
colnames(measureTab) = c("k-means","hclust-single","hclust-complete","hclust-average")

library(knitr) # nice table
kable(measureTab,format = "markdown",digits=2,align="l")
```

|         | k-means | hclust-single | hclust-complete | hclust-average |
|:--------|:--------|:--------------|:----------------|:---------------|
| Purity  | 0.96    | 0.56          | 0.95            | 0.95           |
| Entropy | 0.24    | 1.09          | 0.29            | 0.29           |

### 2.5 Summary

Based on these information, **K-means** outperforms other clustering methods: high purity(high accuracy, a cluster mainly contains only 1 class), low entropy(less disorder). But **complete-link and average-link** hierarchical clustering have the same performance. I don't know which to be placed as the second.

Try to evaluate these two based on Senator x Senator distance.

``` r
set.seed(12312)

# do the clustering with senator x senator distance matrix
complete2s = hclust(rollcall.dist,method='complete')
complete.hc2s = cutree(complete2s,k=2)

average2s = hclust(rollcall.dist,method='average')
average.hc2s = cutree(average2s, k=2)

# code result
hcc.result2 = replace(complete.hc2s, complete.hc2s==2, "Dem")
hcc.result2 = replace(hcc.result2, hcc.result2==1, "Rep")

hca.result2 = replace(average.hc2s, average.hc2s==2, "Dem")
hca.result2 = replace(hca.result2, hca.result2==1, "Rep")

# check membership assignment
print("Complete-link hierarchical clustering result comparison")
```

    ## [1] "Complete-link hierarchical clustering result comparison"

``` r
groupCheck(hcc.result2,party)
```

    ## No. 3 senator MURKOWSKI   from  ALASKA  is a Republican but labeled as Democrat 
    ## No. 37 senator COLLINS     from  MAINE   is a Republican but labeled as Democrat 
    ## No. 64 senator CHIESA      from  NEW JER is a Republican but labeled as Democrat 
    ## 0 Democrats are wrongly labeled as Republicans 
    ## 3 Republicans are wrongly labeled as Democrates

``` r
print("Average-link hierarchical clustering result comparison")
```

    ## [1] "Average-link hierarchical clustering result comparison"

``` r
groupCheck(hca.result2,party)
```

    ## No. 3 senator MURKOWSKI   from  ALASKA  is a Republican but labeled as Democrat 
    ## No. 37 senator COLLINS     from  MAINE   is a Republican but labeled as Democrat 
    ## No. 64 senator CHIESA      from  NEW JER is a Republican but labeled as Democrat 
    ## 0 Democrats are wrongly labeled as Republicans 
    ## 3 Republicans are wrongly labeled as Democrates

Classification results are the same. They should be of the same goodness.
