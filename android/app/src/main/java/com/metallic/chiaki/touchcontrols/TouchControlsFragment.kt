// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL

package com.metallic.chiaki.touchcontrols

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.HapticFeedbackConstants
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.lifecycle.LiveData
import androidx.lifecycle.Observer
import com.metallic.chiaki.databinding.FragmentControlsBinding
import com.metallic.chiaki.common.Preferences
import com.metallic.chiaki.lib.ControllerState
import com.metallic.chiaki.settings.TouchLayoutPrefs
import io.reactivex.Observable
import io.reactivex.rxkotlin.Observables.combineLatest
import io.reactivex.subjects.BehaviorSubject
import io.reactivex.subjects.Subject

private enum class MotionDir { UP, DOWN, LEFT, RIGHT }

abstract class TouchControlsFragment : Fragment() {
    protected var ownControllerState = ControllerState()
        set(value) {
            val diff = field != value
            field = value
            if (diff) ownControllerStateSubject.onNext(ownControllerState)
        }

    protected val ownControllerStateSubject: Subject<ControllerState> =
        BehaviorSubject.create<ControllerState>().also { it.onNext(ownControllerState) }

    protected val controllerStateProxy: Subject<Observable<ControllerState>> =
        BehaviorSubject.create<Observable<ControllerState>>().also { it.onNext(ownControllerStateSubject) }

    val controllerState: Observable<ControllerState> get() = controllerStateProxy.flatMap { it }
    var onScreenControlsEnabled: LiveData<Boolean>? = null
}

class DefaultTouchControlsFragment : TouchControlsFragment() {
    private var _binding: FragmentControlsBinding? = null
    private val binding get() = _binding!!

    // Motion handling
    private val motionHandler = Handler(Looper.getMainLooper())
    // Pulse length (250ms) to let the streamer send multiple updates
    private val MOTION_IMPULSE_MS = 250L
    // how often to re-assert synthetic motion (ms)
    private val MOTION_ASSERT_INTERVAL_MS = 40L
    private val MOTION_REPEAT_INTERVAL_MS = 300L
    private val motionRepeatRunnables = mutableMapOf<MotionDir, Runnable>()

    // Safe, normalized accelerometer impulses (G-force-like)
    private val IMPULSE_ACCEL_SAFE = 5.5f // bumped to 5.5g per request
    // Safe, normalized gyroscope impulses
    private val IMPULSE_GYRO_SAFE = 500.0f // bumped to 500.0f per request

    // Synthetic motion active flag (prevents being zeroed by sensor updates by re-asserting values)
    @Volatile
    private var isSyntheticMotionActive: Boolean = false

    // Runnable used to continuously assert synthetic motion while active
    private val syntheticAssertRunnables = mutableMapOf<MotionDir, Runnable>()

    private val NEUTRAL_GYRO = 0.0f
    private val NEUTRAL_ORIENT_W = 1.0f

    // Fire & Drag - continuous velocity/accumulator aiming model
    private val PUBLISH_INTERVAL_MS = 8L
    private val INACTIVITY_TIMEOUT_MS = 30L
    private val FIRE_SMOOTHING = 0.55f

    // aim velocity state (accumulator, clamped to [-1, 1])
    @Volatile
    private var aimX = 0f
    @Volatile
    private var aimY = 0f
    @Volatile
    private var smoothedX = 0f
    @Volatile
    private var smoothedY = 0f
    @Volatile
    private var lastMoveTime = 0L
    @Volatile
    private var isAimActive = false

    private var publisherRunnable: Runnable? = null

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View =
        FragmentControlsBinding.inflate(inflater, container, false).let {
            _binding = it
            controllerStateProxy.onNext(combineLatest(ownControllerStateSubject, binding.touchpadView.controllerState) { a, b -> a or b })
            it.root
        }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Existing wiring
        binding.dpadView.stateChangeCallback = this::dpadStateChanged
        binding.crossButtonView.buttonPressedCallback = buttonStateChanged(ControllerState.BUTTON_CROSS)
        binding.moonButtonView.buttonPressedCallback = buttonStateChanged(ControllerState.BUTTON_MOON)
        binding.pyramidButtonView.buttonPressedCallback = buttonStateChanged(ControllerState.BUTTON_PYRAMID)
        binding.boxButtonView.buttonPressedCallback = buttonStateChanged(ControllerState.BUTTON_BOX)
        binding.l1ButtonView.buttonPressedCallback = buttonStateChanged(ControllerState.BUTTON_L1)
        binding.r1ButtonView.buttonPressedCallback = buttonStateChanged(ControllerState.BUTTON_R1)
        binding.l3ButtonView.buttonPressedCallback = buttonStateChanged(ControllerState.BUTTON_L3)
        binding.r3ButtonView.buttonPressedCallback = buttonStateChanged(ControllerState.BUTTON_R3)
        binding.optionsButtonView.buttonPressedCallback = buttonStateChanged(ControllerState.BUTTON_OPTIONS)
        binding.shareButtonView.buttonPressedCallback = buttonStateChanged(ControllerState.BUTTON_SHARE)
        binding.psButtonView.buttonPressedCallback = buttonStateChanged(ControllerState.BUTTON_PS)

        binding.l2ButtonView.buttonPressedCallback = { ownControllerState = ownControllerState.copy().apply { l2State = if (it) 255U else 0U } }
        binding.r2ButtonView.buttonPressedCallback = { ownControllerState = ownControllerState.copy().apply { r2State = if (it) 255U else 0U } }

        val quantizeStick = { f: Float -> (Short.MAX_VALUE * f).toInt().toShort() }

        binding.leftAnalogStickView.stateChangedCallback = { ownControllerState = ownControllerState.copy().apply { leftX = quantizeStick(it.x); leftY = quantizeStick(it.y) } }
        binding.rightAnalogStickView.stateChangedCallback = { ownControllerState = ownControllerState.copy().apply { rightX = quantizeStick(it.x); rightY = quantizeStick(it.y) } }

        // Fire & Drag continuous free-drag implementation (velocity accumulator)
        binding.fireDragButton.setListener(object : FireDragView.Listener {
            override fun onHoldStart(startRawX: Float, startRawY: Float) {
                // Start aim gesture
                aimX = 0f
                aimY = 0f
                smoothedX = 0f
                smoothedY = 0f
                lastMoveTime = SystemClock.uptimeMillis()
                isAimActive = true

                // Immediately hold triggers and center stick
                ownControllerState = ownControllerState.copy().apply {
                    l2State = 255U
                    r2State = 255U
                    rightX = 0
                    rightY = 0
                }

                startPublisher()
            }

            override fun onDrag(dx: Float, dy: Float) {
                if (!isAimActive) return
                // Accumulate velocity from relative raw deltas
                val sensitivity = Preferences(requireContext()).fireDragSensitivity
                aimX += dx * sensitivity
                aimY += dy * sensitivity
                // Clamp
                aimX = aimX.coerceIn(-1f, 1f)
                aimY = aimY.coerceIn(-1f, 1f)
                lastMoveTime = SystemClock.uptimeMillis()
                // Do NOT publish here; publisherRunnable is responsible for continuous publishing
            }

            override fun onHoldEnd() {
                // End aim gesture
                isAimActive = false
                cancelPublisher()
                aimX = 0f
                aimY = 0f
                smoothedX = 0f
                smoothedY = 0f
                lastMoveTime = 0L

                ownControllerState = ownControllerState.copy().apply {
                    l2State = 0U
                    r2State = 0U
                    rightX = 0
                    rightY = 0
                }
            }
        })

        // Motion buttons (ensure layout IDs present)
        setupMotionButton(binding.motionUpButton, MotionDir.UP, "motionUpButton")
        setupMotionButton(binding.motionDownButton, MotionDir.DOWN, "motionDownButton")
        setupMotionButton(binding.motionLeftButton, MotionDir.LEFT, "motionLeftButton")
        setupMotionButton(binding.motionRightButton, MotionDir.RIGHT, "motionRightButton")

        // Apply saved positions & visibility
        applySavedLayoutToViews()

        onScreenControlsEnabled?.observe(viewLifecycleOwner, Observer {
            view.visibility = if (it) View.VISIBLE else View.GONE
        })
    }

    private fun startPublisher() {
        if (publisherRunnable != null) return
        publisherRunnable = object : Runnable {
            override fun run() {
                if (!isAimActive) return
                val now = SystemClock.uptimeMillis()
                val age = if (lastMoveTime == 0L) Long.MAX_VALUE else (now - lastMoveTime)
                if (age > INACTIVITY_TIMEOUT_MS) {
                    smoothedX = 0f
                    smoothedY = 0f
                    ownControllerState = ownControllerState.copy().apply {
                        rightX = 0
                        rightY = 0
                        l2State = 255U
                        r2State = 255U
                    }
                } else {
                    smoothedX += (aimX - smoothedX) * FIRE_SMOOTHING
                    smoothedY += (aimY - smoothedY) * FIRE_SMOOTHING
                    val qx = (Short.MAX_VALUE * smoothedX).toInt().toShort()
                    val qy = (Short.MAX_VALUE * smoothedY).toInt().toShort()
                    ownControllerState = ownControllerState.copy().apply {
                        rightX = qx
                        rightY = qy
                        l2State = 255U
                        r2State = 255U
                    }
                }
                motionHandler.postDelayed(this, PUBLISH_INTERVAL_MS)
            }
        }
        motionHandler.post(publisherRunnable!!)
    }

    private fun cancelPublisher() {
        publisherRunnable?.let { motionHandler.removeCallbacks(it) }
        publisherRunnable = null
    }

    override fun onDestroyView() {
        // Ensure no stale callbacks remain
        cancelPublisher()
        super.onDestroyView()
    }

    private fun applySavedLayoutToViews() {
        val mapping = mapOf(
            "motionUpButton" to binding.motionUpButton,
            "motionDownButton" to binding.motionDownButton,
            "motionLeftButton" to binding.motionLeftButton,
            "motionRightButton" to binding.motionRightButton,
            "fireDragButton" to binding.fireDragButton
        )

        val experimentalControlsEnabled = Preferences(requireContext()).experimentalTouchControlsEnabled

        mapping.forEach { (idStr, v) ->
            val savedVisible = TouchLayoutPrefs.getButtonVisible(requireContext(), idStr, true)
            val visible = experimentalControlsEnabled && savedVisible
            v.visibility = if (visible) View.VISIBLE else View.GONE

            val (xFrac, yFrac) = TouchLayoutPrefs.getButtonPos(requireContext(), idStr)
            if (xFrac >= 0f && yFrac >= 0f) {
                v.post {
                    val parent = v.parent as View
                    v.x = xFrac * parent.width
                    v.y = yFrac * parent.height
                }
            }
        }
    }

    private fun setupMotionButton(button: View, dir: MotionDir, idStr: String) {
        button.isClickable = true
        button.isFocusable = true
        button.setOnTouchListener { v, ev ->
            when (ev.action) {
                MotionEvent.ACTION_DOWN -> {
                    // visual + haptic feedback
                    try { v.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY) } catch (_: Throwable) {}
                    v.alpha = 0.6f
                    // start synthetic pulse with continuous assertions
                    startSyntheticMotion(dir)
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    stopSyntheticMotion()
                    v.alpha = 1.0f
                }
            }
            true
        }
    }

    private fun startSyntheticMotion(dir: MotionDir) {
        // If already active, restart timer
        stopSyntheticMotion()
        isSyntheticMotionActive = true

        // Start continuous assertion runnable that re-applies synthetic motion every MOTION_ASSERT_INTERVAL_MS
        val assertRunnable = object : Runnable {
            override fun run() {
                if (!isSyntheticMotionActive) return
                applySyntheticForDir(dir)
                motionHandler.postDelayed(this, MOTION_ASSERT_INTERVAL_MS)
            }
        }
        syntheticAssertRunnables[dir] = assertRunnable
        // run immediately
        motionHandler.post(assertRunnable)

        // Schedule stop after pulse duration
        motionHandler.postDelayed({ stopSyntheticMotion() }, MOTION_IMPULSE_MS)
    }

    private fun stopSyntheticMotion() {
        isSyntheticMotionActive = false
        // remove all assertion runnables
        syntheticAssertRunnables.values.forEach { motionHandler.removeCallbacks(it) }
        syntheticAssertRunnables.clear()
        // also remove repeating pulses
        motionRepeatRunnables.values.forEach { motionHandler.removeCallbacks(it) }
        motionRepeatRunnables.clear()
        // clear motion fields once
        clearMotionFields()
    }

    private fun applySyntheticForDir(dir: MotionDir) {
        // Use safe, normalized magnitudes and assert them into ownControllerState repeatedly
        when (dir) {
            MotionDir.DOWN -> {
                // assert both Y and Z accel negative, and both gyro Y/Z positive (swapped axes)
                setMotionFields(0f, -IMPULSE_ACCEL_SAFE, -IMPULSE_ACCEL_SAFE, NEUTRAL_GYRO, IMPULSE_GYRO_SAFE, IMPULSE_GYRO_SAFE)
            }
            MotionDir.UP -> {
                setMotionFields(0f, IMPULSE_ACCEL_SAFE, IMPULSE_ACCEL_SAFE, NEUTRAL_GYRO, -IMPULSE_GYRO_SAFE, -IMPULSE_GYRO_SAFE)
            }
            MotionDir.LEFT -> {
                setMotionFields(-IMPULSE_ACCEL_SAFE, 0f, -IMPULSE_ACCEL_SAFE, NEUTRAL_GYRO, -IMPULSE_GYRO_SAFE, IMPULSE_GYRO_SAFE)
            }
            MotionDir.RIGHT -> {
                setMotionFields(IMPULSE_ACCEL_SAFE, 0f, IMPULSE_ACCEL_SAFE, NEUTRAL_GYRO, IMPULSE_GYRO_SAFE, -IMPULSE_GYRO_SAFE)
            }
        }
    }

    private fun setMotionFields(accelX: Float, accelY: Float, accelZ: Float, gyroX: Float, gyroY: Float, gyroZ: Float) {
        // When synthetic motion is active, repeatedly set the controller state with these values so they are not overwritten by sensor updates
        ownControllerState = ownControllerState.copy().apply {
            this.accelX = accelX
            this.accelY = accelY
            this.accelZ = accelZ
            this.gyroX = gyroX
            this.gyroY = gyroY
            this.gyroZ = gyroZ
            this.orientX = 0.0f
            this.orientY = 0.0f
            this.orientZ = 0.0f
            this.orientW = NEUTRAL_ORIENT_W
        }
    }

    private fun clearMotionFields() {
        ownControllerState = ownControllerState.copy().apply {
            accelX = 0f
            accelY = 0f
            accelZ = 0f
            gyroX = 0f
            gyroY = 0f
            gyroZ = 0f
        }
    }

    private fun dpadStateChanged(direction: DPadView.Direction?) {
        ownControllerState = ownControllerState.copy().apply {
            buttons = ((buttons
                    and ControllerState.BUTTON_DPAD_LEFT.inv()
                    and ControllerState.BUTTON_DPAD_RIGHT.inv()
                    and ControllerState.BUTTON_DPAD_UP.inv()
                    and ControllerState.BUTTON_DPAD_DOWN.inv())
                    or when (direction) {
                DPadView.Direction.UP -> ControllerState.BUTTON_DPAD_UP
                DPadView.Direction.DOWN -> ControllerState.BUTTON_DPAD_DOWN
                DPadView.Direction.LEFT -> ControllerState.BUTTON_DPAD_LEFT
                DPadView.Direction.RIGHT -> ControllerState.BUTTON_DPAD_RIGHT
                DPadView.Direction.LEFT_UP -> ControllerState.BUTTON_DPAD_LEFT or ControllerState.BUTTON_DPAD_UP
                DPadView.Direction.LEFT_DOWN -> ControllerState.BUTTON_DPAD_LEFT or ControllerState.BUTTON_DPAD_DOWN
                DPadView.Direction.RIGHT_UP -> ControllerState.BUTTON_DPAD_RIGHT or ControllerState.BUTTON_DPAD_UP
                DPadView.Direction.RIGHT_DOWN -> ControllerState.BUTTON_DPAD_RIGHT or ControllerState.BUTTON_DPAD_DOWN
                null -> 0U
            })
        }
    }

    private fun buttonStateChanged(buttonMask: UInt) = { pressed: Boolean ->
        ownControllerState = ownControllerState.copy().apply {
            buttons = if (pressed) buttons or buttonMask else buttons and buttonMask.inv()
        }
    }
}
