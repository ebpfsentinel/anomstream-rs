# AWS `randomcutforest-java` comparison

`java-driver/RcfBench.java` and `scripts/nab/RcfBenchNab.java`
are drivers that feed AWS's `randomcutforest-java` 4.4.0 the
same CSV / NAB inputs as the rcf-rs / rrcf / sklearn runners,
then report identical metrics (inserts/s, scores/s, AUC).

## Prerequisites

- OpenJDK 21 or later (tested on 26). Only `javac` + `java`
  needed.

```bash
sudo apt install -y openjdk-26-jdk
```

## Grab the prebuilt jar

AWS publishes `randomcutforest-core-4.4.0` on Maven Central.
Building from source is not supported on JDK 21+ — the upstream
pom pins `lombok 1.18.30` which does not handle the modern JDK
module layout; bumping Lombok to 1.18.38 in the pom does not
fix it.

```bash
mkdir -p /tmp/aws-rcf
curl -sLo /tmp/aws-rcf/randomcutforest-core-4.4.0.jar \
    https://repo1.maven.org/maven2/software/amazon/randomcutforest/randomcutforest-core/4.4.0/randomcutforest-core-4.4.0.jar
# SHA-256:
#   2e851c82add6d4bcdd13e5cd85fdd091b8a28185fe104775761e8ff6606fd51b
```

## Synthetic-corpus bench

```bash
cd rcf-rs
python3 scripts/external-bench/gen_points.py \
    --n 10000 --dim 16 --seed 2026 > data.csv

JAR=/tmp/aws-rcf/randomcutforest-core-4.4.0.jar
cd scripts/external-bench/java-driver
javac -cp "$JAR" RcfBench.java
java -cp ".:$JAR" RcfBench ../../../data.csv 100 256
```

## NAB corpus bench

```bash
git clone --depth 1 https://github.com/numenta/NAB.git /opt/nab

JAR=/tmp/aws-rcf/randomcutforest-core-4.4.0.jar
cd scripts/nab
javac -cp "$JAR" RcfBenchNab.java
java -cp ".:$JAR" RcfBenchNab /opt/nab
```

## Measured numbers (i7-1370P, JDK 26, cold JVM)

Synthetic (D=16, 10k points, 1 % outliers, 30 % warm):

```
points=10000 dim=16 trees=100 sample=256 warm=3000
  inserts        = 3000, total 767 ms   (3.9k/s)
  scores         = 7000, total 328 ms   (21k/s)
  auc            = 1.000
```

NAB `realKnownCause` aggregate weighted AUC: **0.757**.

## Notes

- Numbers are **cold JVM** — no JMH warmup. A proper JVM
  micro-benchmark would warm JIT for 5–10 s before measuring.
  The cold numbers here represent a realistic process-startup
  cost for a shell-invoked job, which is the fair comparison
  against a native Rust binary.
- AWS Java's `getAnomalyScore` uses a probability-of-separation
  visitor (probe-based), closer to `rrcf`'s `codisp` than to
  `rcf-rs`'s isolation-depth `score()` — this drives the ~0.14
  NAB AUC gap (AWS 0.757, rrcf 0.748, rcf-rs 0.615).
