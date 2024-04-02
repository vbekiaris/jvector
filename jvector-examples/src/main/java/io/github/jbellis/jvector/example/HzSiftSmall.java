package io.github.jbellis.jvector.example;

import com.hazelcast.config.Config;
import com.hazelcast.config.vector.Metric;
import com.hazelcast.config.vector.VectorCollectionConfig;
import com.hazelcast.config.vector.VectorIndexConfig;
import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.vector.SearchOptions;
import com.hazelcast.vector.VectorCollection;
import com.hazelcast.vector.VectorDocument;
import com.hazelcast.vector.VectorValues;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.pq.CompressedVectors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class HzSiftSmall {

    private static final String VECTOR_COLLECTION_NAME = "sift-small";

    public static void testRecall(ArrayList<float[]> baseVectors, ArrayList<float[]> queryVectors, ArrayList<HashSet<Integer>> groundTruth, Path testDirectory) throws IOException, InterruptedException, ExecutionException {
        int originalDimension = baseVectors.get(0).length;
        int size = baseVectors.size();
        var ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        Config config = new Config();
        VectorCollectionConfig vcc = new VectorCollectionConfig(VECTOR_COLLECTION_NAME);
        vcc.addVectorIndexConfig(new VectorIndexConfig("sift-small-index", Metric.EUCLIDEAN, originalDimension));
        config.addVectorCollectionConfig(vcc);
        HazelcastInstance member = Hazelcast.newHazelcastInstance(config);
        VectorCollection<Integer, Integer> vectorCollection = VectorCollection.getCollection(member, VECTOR_COLLECTION_NAME);

        var start = System.nanoTime();
        IntStream.range(0, size).forEach(i -> {
            vectorCollection.setAsync(i, VectorDocument.of(i, VectorValues.of(baseVectors.get(i)))).toCompletableFuture().join();
            if (i % 1000 == 0) {
                System.out.println("Indexed " + i + " items");
            }
        });
        System.out.printf("  Building index took %s seconds%n", (System.nanoTime() - start) / 1_000_000_000.0);
        testRecallInternal(vectorCollection, ravv, queryVectors, groundTruth, null);
        member.shutdown();
    }

    private static void testRecallInternal(VectorCollection<Integer, Integer> vectorCollection, RandomAccessVectorValues<float[]> ravv, ArrayList<float[]> queryVectors, ArrayList<HashSet<Integer>> groundTruth, CompressedVectors compressedVectors) {
        var topKfound = new AtomicInteger(0);
        var topK = 100;
        var start = System.nanoTime();
        IntStream.range(0, queryVectors.size()).parallel().forEach(i -> {
            var queryVector = queryVectors.get(i);
            ArrayList<Integer> resultNodes = new ArrayList<>(110);
            var searchResults = vectorCollection.searchAsync(SearchOptions.of(queryVector, 100, false, false)).toCompletableFuture().join();
            searchResults.results().forEachRemaining(result -> {
                resultNodes.add((Integer) result.getKey());
            });

            var gt = groundTruth.get(i);
            var n = IntStream.range(0, topK).filter(j -> gt.contains(resultNodes.get(j))).count();
            topKfound.addAndGet((int) n);
        });
        System.out.printf("Querying %d vectors in parallel took %s seconds%n", queryVectors.size(), (System.nanoTime() - start) / 1_000_000_000.0);
        System.out.printf("Recall: %.4f%n", (double) topKfound.get() / (queryVectors.size() * topK));
    }

    public static void main(String[] args) throws IOException, ExecutionException, InterruptedException {
        var siftPath = "siftsmall";
        var baseVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_base.fvecs", siftPath));
        var queryVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_query.fvecs", siftPath));
        var groundTruth = SiftLoader.readIvecs(String.format("%s/siftsmall_groundtruth.ivecs", siftPath));
        System.out.format("%d base and %d query vectors loaded, dimensions %d%n",
                baseVectors.size(), queryVectors.size(), baseVectors.get(0).length);

        var testDirectory = Files.createTempDirectory("SiftSmallGraphDir");
        try {
            testRecall(baseVectors, queryVectors, groundTruth, testDirectory);
        } finally {
            Files.delete(testDirectory);
        }
    }
}
