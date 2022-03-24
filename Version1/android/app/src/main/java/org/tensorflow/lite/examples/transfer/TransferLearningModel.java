/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.transfer;

import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/** Represents a "partially" trainable model that is based on some other, base model. */
public final class TransferLearningModel implements Closeable {

  /**
   * Prediction for a single class produced by the model.
   */
  public static int SAMPLE_NUM = 0;

  public static class Prediction {
    private final String className;
    private final float confidence;

    public Prediction(String className, float confidence) {
      this.className = className;
      this.confidence = confidence;
    }

    public String getClassName() {
      return className;
    }

    public float getConfidence() {
      return confidence;
    }
  }
  private static class TrainingSample {
    float[] bottleneck;
    float[] label;

    TrainingSample(float[] bottleneck, float[] label) {
      this.bottleneck = bottleneck;
      this.label = label;
    }
  }

  /**
   * Consumer interface for training loss.
   */
  public interface LossConsumer { //epoch, loss 를 가짐
    void onLoss(int epoch, float loss);
  }

  // Setting this to a higher value allows to calculate bottlenecks for more samples while
  // adding them to the bottleneck collection is blocked by an active training thread.
  private static final int NUM_THREADS =  //1, 사용 가능한 코어 수 - 1 중에 최대값
          Math.max(1, Runtime.getRuntime().availableProcessors() - 1);

  private final Map<String, Integer> classes;
  private final String[] classesByIdx;
  private final Map<String, float[]> oneHotEncodedClass;

  private final LiteMultipleSignatureModel model;

  private final List<TrainingSample> trainingSamples = new ArrayList<>();

  // Where to store training inputs.
  private float[][] trainingBatchBottlenecks;
  private float[][] trainingBatchLabels;

  // Used to spawn background threads.
  private final ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS); //7개의 쓰레드풀 생성

  // This lock guarantees that only one thread is performing training and inference at
  // any point in time. It also protects the sample collection from being modified while
  // in use by a training thread.
  private final Lock trainingInferenceLock = new ReentrantLock();

  // This lock guards access to trainable parameters.
  private final ReadWriteLock parameterLock = new ReentrantReadWriteLock();

  // Set to true when [close] has been called.
  private volatile boolean isTerminating = false;

  public TransferLearningModel(ModelLoader modelLoader, Collection<String> classes) {
    try {
      this.model =  //모델로더에서 모델 읽기
              new LiteMultipleSignatureModel( //새로운 모델 생성, 클래스 갯수 지정
                      modelLoader.loadMappedFile("model.tflite"), classes.size());
    } catch (IOException e) {
      throw new RuntimeException("Couldn't read underlying model for TransferLearningModel", e);
    }
    classesByIdx = classes.toArray(new String[0]);  //클래스 순서 배열
    this.classes = new TreeMap<>(); //클래스 맵
    oneHotEncodedClass = new HashMap<>(); //One_hot_Encoding 기법을 위한 HashMap - ex: [0, 0, 1, 0] -> 3번째 클래스 hot
    for (int classIdx = 0; classIdx < classes.size(); classIdx++) { //클래스 순서 0 ~ 최대 수 까지
      String className = classesByIdx[classIdx];  //클래스 이름 = 클래스 순서대로 get
      this.classes.put(className, classIdx);    //클래스 맵에 이름, 순서 추가
      oneHotEncodedClass.put(className, oneHotEncoding(classIdx));  //One_hot_Encoding 해시맵에 0, [1, 0, 0, 0]
      //                         1, [0, 1, 0, 0]
      //                         2, [0, 0, 1, 0]
      //                         3, [0, 0, 0, 1]  4가지 추가
    }
  }

  /**
   * Adds a new sample for training.
   *
   * <p>Sample bottleneck is generated in a background thread, which resolves the returned Future
   * when the bottleneck is added to training samples.
   *
   * @param image image RGB data.
   * @param className ground truth label for image.
   */
  @RequiresApi(api = Build.VERSION_CODES.O)
  public Future<Void> addSample(float[][][] image, String className) {  //샘플 추가
    checkNotTerminating();  //종료되었는지 확인

    if (!classes.containsKey(className)) {  //클래스 이름이 일치하지 않으면 throw
      throw new IllegalArgumentException(String.format(
              "Class \"%s\" is not one of the classes recognized by the model", className));
    }

    return executor.submit( //addSample.get() 시에 반환받음
            () -> {
              if (Thread.interrupted()) { //thread 가 종료되면 return null
                return null;
              }

              trainingInferenceLock.lockInterruptibly();//interrupt 가능한 lock
              try {
                float[] bottleneck = model.loadBottleneck(image); //이미지의 bottleneck 추출 float[62720]

//            ---------- 파일 쓰기----------
//          샘플 수 1700 이하면 기존의 방식 사용
                if (SAMPLE_NUM >= 1700) {
                  //파일 읽기
                  File file = new File(MainActivity.dir+"/sample.data");
                  RandomAccessFile randomAccessFile = new RandomAccessFile(file, "rw");
                  //파일 채널 생성
                  FileChannel fileChannel = randomAccessFile.getChannel();
                  fileChannel.position((SAMPLE_NUM - 1700) * 62724 * 4L);

                  //bottleneck 추가
                  ByteBuffer buffer = ByteBuffer.allocate(62720 * 4);
                  buffer.clear();
                  buffer.rewind();
                  buffer.asFloatBuffer().put(bottleneck);
                  fileChannel.write(buffer);

                  //oneHotEncodedClass 추가

                  ByteBuffer buffer1 = ByteBuffer.allocate(4 * 4);
                  buffer1.clear();
                  buffer1.rewind();
                  buffer1.asFloatBuffer().put(oneHotEncodedClass.get(className));
                  fileChannel.write(buffer1);
                  fileChannel.close();

                } else {
                  trainingSamples.add(new TrainingSample(bottleneck, oneHotEncodedClass.get(className))); //훈련 샘플 추가, bottleneck, on hot encoding
                }
                SAMPLE_NUM++; //샘플 수 증가

              } catch (Exception e) {
                e.printStackTrace();
              }
              finally {
                trainingInferenceLock.unlock();
              }

              return null;  //return null
            });
  }

  /**
   * Trains the model on the previously added data samples.
   *
   * @param numEpochs number of epochs to train for.
   * @param lossConsumer callback to receive loss values, may be null.
   * @return future that is resolved when training is finished.
   */
  @RequiresApi(api = Build.VERSION_CODES.O)
  public Future<Void> train(int numEpochs, LossConsumer lossConsumer) {
    checkNotTerminating();  //종료되었는지 확인
    int trainBatchSize = getTrainBatchSize(); //훈련 배치 사이즈 가져오기

    if (SAMPLE_NUM < trainBatchSize) {  //샘플이 배치 사이즈보다 적으면
      throw new RuntimeException(                   //== 샘플이 0개이고 배치 사이즈가 1인 경우
              String.format(
                      "Too few samples to start training: need %d, got %d",
                      trainBatchSize, SAMPLE_NUM));
    }

    trainingBatchBottlenecks = new float[trainBatchSize][numBottleneckFeatures()];  //배치 사이즈 * bottlenecks 크기
    trainingBatchLabels = new float[trainBatchSize][this.classes.size()]; //배치 사이즈 * 라벨

    return executor.submit( //train.get() 을 호출하면 반환
            () -> {
              trainingInferenceLock.lock(); //lock
              try {
                epochLoop:  //go to 구문  //다시 실
                for (int epoch = 0; epoch < numEpochs; epoch++) { //에폭 수 만큼 for 루프
                  float totalLoss = 0;  //총 loss 값
                  int numBatchesProcessed = 0;  //배치 프로세스 수

                  List<Integer> shuffle = new ArrayList<>();
                  if (SAMPLE_NUM > 1700) {
                    for (int i = 0; i < (SAMPLE_NUM - 1700); i ++) {
                      shuffle.add(i);
                    }
                  }
                  Collections.shuffle(shuffle);

                  long startTime = System.currentTimeMillis();
                  for (List<TrainingSample> batch : trainingBatches(trainBatchSize, shuffle)) {  //for (A : B) 는 B의 값이 없을 때 까지 B에서 하나씩 꺼내서 A에게 넣는다는 의미이다.
                    //훈련 배치 사이즈만큼 (ex: 20)
                    if (Thread.interrupted()) { //쓰레드가 중단되었을 때
                      break epochLoop; //go to 구문
                    }
//                    Log.d("?!", Arrays.toString(batch.get(0).label) + Arrays.toString(batch.get(0).bottleneck));
//                    Log.d("?!", Arrays.toString(batch.get(1).label) + Arrays.toString(batch.get(1).bottleneck));
//                    Log.d("?!", Arrays.toString(batch.get(2).label) + Arrays.toString(batch.get(2).bottleneck));
//                    Log.d("?!", Arrays.toString(batch.get(3).label) + Arrays.toString(batch.get(3).bottleneck));
//                    Log.d("?!", Arrays.toString(batch.get(4).label) + Arrays.toString(batch.get(4).bottleneck));
                      for (int sampleIdx = 0; sampleIdx < batch.size(); sampleIdx++) {  //배치 사이즈까지 반복 (ex: 20)
                      TrainingSample sample = batch.get(sampleIdx);   //샘플 = trainingBatches 에서 가져온 샘플
                      trainingBatchBottlenecks[sampleIdx] = sample.bottleneck;  //bottleneck, label 설정
                      trainingBatchLabels[sampleIdx] = sample.label;
                    }

                    float loss = this.model.runTraining(trainingBatchBottlenecks, trainingBatchLabels); //loss 는 학습시킨 결과, bottleneck, label 투입
                    totalLoss += loss;  //totalLoss add
                    numBatchesProcessed++;  //처리한 배치 수 add
                  }

                  float avgLoss = totalLoss / numBatchesProcessed;  //totalLoss / 배치 수
                  if (lossConsumer != null) { //lossConsumer null 이면
                    lossConsumer.onLoss(epoch, avgLoss);  //lossConsumer 에 epoch, Loss 설정 ->  ViewModel.setLastLoss(loss)
                  }
                  long endTime = System.currentTimeMillis();
                  Log.d("1 epoch 학습시간: ", endTime - startTime + "\n" +
                          "Loss: " + avgLoss);
                }

                return null;
              } finally {
                trainingInferenceLock.unlock(); //락 해제
              }
            });
  }

  /**
   * Runs model inference on a given image.
   *
   * @param image image RGB data.
   * @return predictions sorted by confidence decreasing. Can be null if model is terminating.
   */
  public Prediction[] predict(float[][][] image) {
    checkNotTerminating();
    trainingInferenceLock.lock();

    try {
      if (isTerminating) {
        return null;
      }

      float[] confidences;
      parameterLock.readLock().lock();
      try {
        confidences = this.model.runInference(image);
      } finally {
        parameterLock.readLock().unlock();
      }

      Prediction[] predictions = new Prediction[classes.size()];
      for (int classIdx = 0; classIdx < classes.size(); classIdx++) {
        predictions[classIdx] = new Prediction(classesByIdx[classIdx], confidences[classIdx]);
      }

      Arrays.sort(predictions, (a, b) -> -Float.compare(a.confidence, b.confidence));
      return predictions;
    } finally {
      trainingInferenceLock.unlock();
    }
  }

  private float[] oneHotEncoding(int classIdx) {
    float[] oneHot = new float[4];
    oneHot[classIdx] = 1;
    return oneHot;
  }

  /** Training model expected batch size. */
  public int getTrainBatchSize() {  //훈련 배치 사이즈
    return Math.min(  //1 ~ 훈련 샘플 사이즈 중 최대값과, expectedBatchSize( = 20)중 최소값 리턴  -> 최소 1장의 훈련 샘플 필요
            Math.max(/* at least one sample needed */ 1, SAMPLE_NUM),
            model.getExpectedBatchSize());
  }

  /**
   * Constructs an iterator that iterates over training sample batches.
   *
   * @param trainBatchSize batch size for training.
   * @return iterator over batches.
   */
  private Iterable<List<TrainingSample>> trainingBatches(int trainBatchSize, List<Integer> shuffle) {
    if (!trainingInferenceLock.tryLock()) { //lock 시도 실패시 Throw
      throw new RuntimeException("Thread calling trainingBatches() must hold the training lock");
    }
    trainingInferenceLock.unlock(); //lock 해제

    Collections.shuffle(trainingSamples); //training Samples 랜덤하게 셔플
    return () ->
            new Iterator<List<TrainingSample>>() {
              private int nextIndex = 0;

              @RequiresApi(api = Build.VERSION_CODES.O)
              @Override
              public boolean hasNext() {
                return nextIndex < SAMPLE_NUM;  //샘플이 남아 있으면 true
              }

              @RequiresApi(api = Build.VERSION_CODES.O)
              @Override
              public List<TrainingSample> next() {
                int fromIndex = nextIndex;  //시작 index ex: 0
                int toIndex = nextIndex + trainBatchSize; //목표 index = nextIndex + 배치 사이즈 ex: 20
                nextIndex = toIndex;  //nextIndex 업데이트 ex: 20

                List<TrainingSample> trainingSamples_file = new ArrayList<>(); //샘플 리턴 위한 list
                if (toIndex >= SAMPLE_NUM) {  //목표 인덱스보다 훈련 샘플이 작을 때
                  if (fromIndex < 1700) {
                    return trainingSamples.subList(
                            trainingSamples.size() - trainBatchSize, trainingSamples.size());
                  } else {
                    try {
                      for (int i = SAMPLE_NUM - 1; i >= fromIndex; i --) {
                        File file = new File(MainActivity.dir + "/sample.data");
                        RandomAccessFile randomAccessFile = new RandomAccessFile(file, "r");
                        FileChannel fileChannel = randomAccessFile.getChannel();

                        fileChannel.position(shuffle.get(i-1700) * 62724 * 4L);
                        float[] bottleneck_file = new float[62720];
                        float[] oneHotEncoded_file = new float[4];
                        
                        ByteBuffer buffer = ByteBuffer.allocate(62720 * 4);
                        buffer.clear();
                        fileChannel.read(buffer);
                        buffer.rewind();
                        buffer.asFloatBuffer().get(bottleneck_file);
                        
                        ByteBuffer buffer1 = ByteBuffer.allocate(4 * 4);
                        buffer1.clear();
                        fileChannel.read(buffer1);
                        buffer1.rewind();
                        buffer1.asFloatBuffer().get(oneHotEncoded_file);
                        
                        trainingSamples_file.add(new TrainingSample(bottleneck_file, oneHotEncoded_file));
                        fileChannel.close();
                      }
                    } catch (Exception e) {
                      e.printStackTrace();
                    }
                    return trainingSamples_file;
                  }
                } else {
                  if (fromIndex < 1700) {
                    return trainingSamples.subList(fromIndex, toIndex);
                  } else {
                    try {
                      for (int i = 0; i < trainBatchSize; i ++) {
                        File file = new File(MainActivity.dir + "/sample.data");
                        RandomAccessFile randomAccessFile = new RandomAccessFile(file, "r");
                        FileChannel fileChannel = randomAccessFile.getChannel();

                        fileChannel.position(shuffle.get(fromIndex + i -1700) * 62724 * 4L);
                        float[] bottleneck_file = new float[62720];
                        float[] oneHotEncoded_file = new float[4];

                        ByteBuffer buffer = ByteBuffer.allocate(62720 * 4);
                        buffer.clear();
                        fileChannel.read(buffer);
                        buffer.rewind();
                        buffer.asFloatBuffer().get(bottleneck_file);

                        ByteBuffer buffer1 = ByteBuffer.allocate(4 * 4);
                        buffer1.clear();
                        fileChannel.read(buffer1);
                        buffer1.rewind();
                        buffer1.asFloatBuffer().get(oneHotEncoded_file);

                        trainingSamples_file.add(new TrainingSample(bottleneck_file, oneHotEncoded_file));
                        fileChannel.close();
                      }
                    } catch (Exception e) {
                      e.printStackTrace();
                    }
                    return trainingSamples_file;
                  }
                }
              }
            };
  }

  private int numBottleneckFeatures() { //bottleneckFeatures 수 리턴
    return model.getNumBottleneckFeatures();  //train 할 때 InputTensor 를 get 함
  }

  private void checkNotTerminating() {
    if (isTerminating) {  //isTerminating 이 true 면 throw
      throw new IllegalStateException("Cannot operate on terminating model");
    }
  }

  /**
   * Terminates all model operation safely. Will block until current inference request is finished
   * (if any).
   *
   * <p>Calling any other method on this object after [close] is not allowed.
   */
  @Override
  public void close() {
    isTerminating = true;
    executor.shutdownNow();

    // Make sure that all threads doing inference are finished.
    trainingInferenceLock.lock();

    try {
      boolean ok = executor.awaitTermination(5, TimeUnit.SECONDS);
      if (!ok) {
        throw new RuntimeException("Model thread pool failed to terminate");
      }

      this.model.close();
    } catch (InterruptedException e) {
      // no-op
    } finally {
      trainingInferenceLock.unlock();
    }
  }
}
