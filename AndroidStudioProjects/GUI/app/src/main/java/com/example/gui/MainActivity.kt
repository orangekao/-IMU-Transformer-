package com.example.gui
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothSocket
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.util.*
import android.graphics.Typeface
//繪圖
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.github.mikephil.charting.components.Legend
import com.github.mikephil.charting.components.XAxis

import org.json.JSONObject
import org.json.JSONArray

import android.content.Intent
import android.os.Handler
import android.os.Looper

//燈
import android.view.View
//(json)
//傳進來的是：'inference':
//          'probability':
//          'data':

// 載入畫面
class SplashActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_splash)

        // 延遲 2 秒跳轉到 MainActivity
        Handler(Looper.getMainLooper()).postDelayed({
            startActivity(Intent(this, MainActivity::class.java))
            finish()
        }, 4000) // 可調整秒數
    }
}
// MainActivity繼承使用者界面(UI)AppCompatActivity()
class MainActivity : AppCompatActivity() {
    private lateinit var inferenceResult: TextView // 接收的訊息
    private lateinit var signalChart: LineChart
    private lateinit var probability: TextView
    private lateinit var signal: TextView // 接收的訊息
    private val bluetoothAdapter: BluetoothAdapter? = BluetoothAdapter.getDefaultAdapter() // 藍芽模組
    private val TAG = "BluetoothTest"
    private val PI_MAC = "AA:AA:AA:AA:AA:AA" // Pi MAC
//    private val PI_MAC = "50:84:92:9D:44:36" // Pc MAC
    private val MY_UUID: UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB") // SPP固定UUID
    private lateinit var toggleButton: Button
    private var isConnected = false
    private var connectThread: ConnectThread? = null
    private var bluetoothSocket: BluetoothSocket? = null
    private var isReading = false
    private lateinit var statusLight: View

    private fun connectToPi()
    {
        toggleButton.text = "Connecting..."
        statusLight.setBackgroundResource(R.drawable.status_light_orange)
        @Suppress("MissingPermission")
        bluetoothAdapter?.cancelDiscovery()

        val device = bluetoothAdapter?.getRemoteDevice(PI_MAC)
        if(device != null)
        {
            connectThread = ConnectThread(device)
            connectThread?.start()
        }
        else
        {
            toggleButton.text = "Press to connect!"
            statusLight.setBackgroundResource(R.drawable.status_light_red)
        }

    }

    private fun disconnectToPi()
    {
        try
        {
            connectThread?.cancel()
            bluetoothSocket?.close()
            isConnected = false
            toggleButton.text = "Press to connect!" // 斷線後更新按鈕狀態
            statusLight.setBackgroundResource(R.drawable.status_light_red)
            // 斷線後顯示 default_image
            val imageView = findViewById<de.hdodenhof.circleimageview.CircleImageView>(R.id.profile_image)
            imageView.setImageResource(R.drawable.default_image)
        } catch (_: IOException) {
            toggleButton.text = "Press to connect!" // 出錯也一律回到可連線狀態
            statusLight.setBackgroundResource(R.drawable.status_light_red)
        }
    }




    // override fun可幫助偵錯，onCreate是模組名稱, 多加的元件都要在這裡綁定
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        //這個一定要先不然會閃退（指定畫面 layout）
        setContentView(R.layout.activity_main)

        // 啟動但為連線時顯示default_image
        val imageView = findViewById<de.hdodenhof.circleimageview.CircleImageView>(R.id.profile_image)
        imageView.setImageResource(R.drawable.default_image)

        // 綁定 UI 元件
        inferenceResult = findViewById(R.id.inferenceResult)
        statusLight = findViewById(R.id.connectionStatusLight)

        probability = findViewById(R.id.probability)
        signalChart = findViewById(R.id.signalChart)
        toggleButton = findViewById(R.id.toggleButton)
        toggleButton.text = "Press to connect!"
        inferenceResult.text = null
        probability.text = null

        // 可選：初始化圖表樣式
        signalChart.description.isEnabled = false
        signalChart.setTouchEnabled(false)
        signalChart.setPinchZoom(false)
        signalChart.legend.isEnabled = true

        // 按鈕點擊事件設定
        toggleButton.setOnClickListener {
            if (isConnected) {
                disconnectToPi()
            } else {
                connectToPi()
            }
        }
    }

    // 啟動畫面 → 顯示初始訊息 → 透過藍牙 MAC 嘗試連線 Pi → 更新 UI 狀態


    private fun manageMyConnectedSocket(socket: BluetoothSocket) {

        //機率表格用
        val classes = listOf("Sitting", "Fall", "Sit_down", "Stand_up", "Walking", "Walk_stairs", "Push_up", "Jumping")
        val reader = BufferedReader(InputStreamReader(socket.inputStream))
        isReading = true

        Thread {
            while (isReading) {
                try {
                    val line = reader.readLine() ?: break
                    val json = JSONObject(line)

                    val inference = json.getString("inference")
                    val probArray = json.getJSONArray("probability")
                    val signalArray = json.getJSONArray("data")

                    // 將 JSON array 轉成 List<Double>
                    val probList = (0 until probArray.length()).map { probArray.getDouble(it) }

                    // 組裝顯示字串
                    val displayText = buildString {
                        for (i in classes.indices) {
                            val probValue = probList.getOrNull(i) ?: 0.0
                            append(String.format("%-12s %.2f\n\n", "${classes[i]}:", probValue))
                        }
                    }

                    //繪製signal
                    // 初始化三個軸的資料
                    val signalX = mutableListOf<Float>()
                    val signalY = mutableListOf<Float>()
                    val signalZ = mutableListOf<Float>()
//
//                    // 分別取出xyz
                    for (i in 0 until signalArray.length()) {
                        val triple = signalArray.getJSONArray(i)
                        signalX.add(triple.getDouble(0).toFloat())
                        signalY.add(triple.getDouble(1).toFloat())
                        signalZ.add(triple.getDouble(2).toFloat())
                    }

                    // 繪製
                    val entriesX = signalX.mapIndexed { index, value -> Entry(index.toFloat(), value) }
                    val entriesY = signalY.mapIndexed { index, value -> Entry(index.toFloat(), value) }
                    val entriesZ = signalZ.mapIndexed { index, value -> Entry(index.toFloat(), value) }

                    val dataSetX = LineDataSet(entriesX, "X").apply {
                        color = android.graphics.Color.RED
                        setDrawCircles(false)
                        lineWidth = 2f
                    }

                    val dataSetY = LineDataSet(entriesY, "Y").apply {
                        color = android.graphics.Color.GREEN
                        setDrawCircles(false)
                        lineWidth = 2f
                    }

                    val dataSetZ = LineDataSet(entriesZ, "Z").apply {
                        color = android.graphics.Color.BLUE
                        setDrawCircles(false)
                        lineWidth = 2f
                    }


                    runOnUiThread {
                        signalChart.setExtraOffsets(0f, 0f, 0f, 10f)
                        signalChart.data = LineData(dataSetX, dataSetY, dataSetZ)

                        // 圖表更新
                        signalChart.invalidate()
                        // 設定圖例
                        signalChart.legend.apply {
                            isEnabled = true
                            verticalAlignment = Legend.LegendVerticalAlignment.TOP
                            horizontalAlignment = Legend.LegendHorizontalAlignment.CENTER
                            orientation = Legend.LegendOrientation.HORIZONTAL
                            setDrawInside(false)
                            textSize = 12f
                        }

                        // 設定 X 軸（在下方）
                        signalChart.xAxis.apply {
                            position = XAxis.XAxisPosition.BOTTOM
                            setDrawGridLines(false)
                            granularity = 1f
                            labelRotationAngle = 0f
                            textSize = 12f
                        }

                        // 設定 Y 軸（只顯示左邊）
                        signalChart.axisLeft.apply {
                            isEnabled = true
                            textSize = 12f
                        }
                        signalChart.axisRight.isEnabled = false

                        inferenceResult.text = inference
                        probability.typeface = Typeface.MONOSPACE
                        probability.text = displayText

                        // 繪製信號
                        signalChart.data = LineData(dataSetX, dataSetY, dataSetZ)
                        signalChart.invalidate() // 重新繪製圖表

                        // 顯示對應圖片
                        val imageView = findViewById<de.hdodenhof.circleimageview.CircleImageView>(R.id.profile_image)
                        // 對照inference的輸出，一律為小寫且移除空格改為底限
                        val imageName = inference.lowercase().replace(" ", "_")
                        val resId = resources.getIdentifier(imageName, "drawable", packageName)
                        imageView.setImageResource(if (resId != 0) resId else R.drawable.default_image)

                    }

                } catch (e: Exception) {
                    e.printStackTrace()
                    break
                }
            }
        }.start()
    }



    //開啟背景連線，防止app卡死
    private inner class ConnectThread(device: BluetoothDevice) : Thread() {

        private val mmDevice = device
        private var socket: BluetoothSocket? = null

        override fun run() {
            try {
                @Suppress("MissingPermission")
                bluetoothAdapter?.cancelDiscovery()

                @Suppress("MissingPermission")
                socket = mmDevice.createRfcommSocketToServiceRecord(MY_UUID)

                @Suppress("MissingPermission")
                socket?.connect()
                bluetoothSocket = socket
                isConnected = true

                runOnUiThread {
                    statusLight.setBackgroundResource(R.drawable.status_light_green)
                    toggleButton.text = "Press to disconnect!"
                }

                socket?.let { manageMyConnectedSocket(it) }


            } catch (e: IOException) {
                isConnected = false
                runOnUiThread {
                    statusLight.setBackgroundResource(R.drawable.status_light_red)
                    toggleButton.text = "Press to connect!"
                }

                try {
                    socket?.close()
                } catch (_: IOException) {}
            }
        }

        fun cancel() {
            try {
                socket?.close()
                isConnected = false
                runOnUiThread {
                    statusLight.setBackgroundResource(R.drawable.status_light_red)
                    toggleButton.text = "Press to connect!"
                    inferenceResult.text = null
                    probability.text = null
                    val imageView = findViewById<de.hdodenhof.circleimageview.CircleImageView>(R.id.profile_image)
                    imageView.setImageResource(R.drawable.default_image)
                    signalChart.clear()
                    signalChart.invalidate()

                }
            } catch (_: IOException) {}
        }
    }

}
