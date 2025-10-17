package ai.baseweight.baseweightsnap.ui

import ai.baseweight.baseweightsnap.R
import ai.baseweight.baseweightsnap.models.HFModelMetadata
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import java.text.SimpleDateFormat
import java.util.*

class ModelAdapter(
    private val onModelClick: (HFModelMetadata) -> Unit,
    private val onSetDefault: (HFModelMetadata) -> Unit,
    private val onDelete: (HFModelMetadata) -> Unit,
    private val onCancelDownload: (HFModelMetadata) -> Unit
) : ListAdapter<HFModelMetadata, ModelAdapter.ModelViewHolder>(ModelDiffCallback()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ModelViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_model, parent, false)
        return ModelViewHolder(view)
    }

    override fun onBindViewHolder(holder: ModelViewHolder, position: Int) {
        val model = getItem(position)
        holder.bind(model)
    }

    inner class ModelViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val ivDefaultIndicator: ImageView = itemView.findViewById(R.id.ivDefaultIndicator)
        private val tvModelName: TextView = itemView.findViewById(R.id.tvModelName)
        private val tvRepository: TextView = itemView.findViewById(R.id.tvRepository)
        private val tvSize: TextView = itemView.findViewById(R.id.tvSize)
        private val tvDownloadDate: TextView = itemView.findViewById(R.id.tvDownloadDate)
        private val btnMoreOptions: ImageButton = itemView.findViewById(R.id.btnMoreOptions)

        fun bind(model: HFModelMetadata) {
            // Show/hide default indicator
            ivDefaultIndicator.visibility = if (model.isDefault) View.VISIBLE else View.GONE

            // Set model name
            tvModelName.text = model.name

            // Set repository
            tvRepository.text = model.hfRepo

            // Handle different download states (handle null for backwards compatibility)
            when (model.downloadState ?: ai.baseweight.baseweightsnap.models.DownloadState.COMPLETED) {
                ai.baseweight.baseweightsnap.models.DownloadState.PENDING,
                ai.baseweight.baseweightsnap.models.DownloadState.DOWNLOADING -> {
                    tvSize.text = "Downloading: ${model.downloadProgress}%"
                    tvDownloadDate.text = "Download in progress..."
                    btnMoreOptions.isEnabled = true
                    btnMoreOptions.setImageResource(android.R.drawable.ic_menu_close_clear_cancel)
                    itemView.alpha = 0.7f
                }
                ai.baseweight.baseweightsnap.models.DownloadState.ERROR -> {
                    tvSize.text = "Download failed"
                    tvDownloadDate.text = "Error downloading model"
                    btnMoreOptions.isEnabled = true
                    btnMoreOptions.setImageResource(android.R.drawable.ic_menu_more)
                    itemView.alpha = 1.0f
                }
                ai.baseweight.baseweightsnap.models.DownloadState.COMPLETED -> {
                    tvSize.text = "Size: ${model.formatSize()}"
                    val dateFormat = SimpleDateFormat("MMM dd, yyyy", Locale.getDefault())
                    tvDownloadDate.text = "Downloaded: ${dateFormat.format(Date(model.downloadDate))}"
                    btnMoreOptions.isEnabled = true
                    btnMoreOptions.setImageResource(android.R.drawable.ic_menu_more)
                    itemView.alpha = 1.0f
                }
            }

            // Click listener for the whole item
            itemView.setOnClickListener {
                onModelClick(model)
            }

            // More options menu - or cancel button for downloading models
            btnMoreOptions.setOnClickListener { view ->
                if (model.downloadState == ai.baseweight.baseweightsnap.models.DownloadState.PENDING ||
                    model.downloadState == ai.baseweight.baseweightsnap.models.DownloadState.DOWNLOADING) {
                    // Cancel download
                    onCancelDownload(model)
                } else {
                    // Show options menu
                    showPopupMenu(view, model)
                }
            }
        }

        private fun showPopupMenu(view: View, model: HFModelMetadata) {
            val popup = androidx.appcompat.widget.PopupMenu(view.context, view)
            popup.menuInflater.inflate(R.menu.model_options_menu, popup.menu)

            // Disable "Set as Default" if already default
            popup.menu.findItem(R.id.action_set_default)?.isEnabled = !model.isDefault

            popup.setOnMenuItemClickListener { menuItem ->
                when (menuItem.itemId) {
                    R.id.action_set_default -> {
                        onSetDefault(model)
                        true
                    }
                    R.id.action_delete -> {
                        onDelete(model)
                        true
                    }
                    else -> false
                }
            }
            popup.show()
        }
    }

    class ModelDiffCallback : DiffUtil.ItemCallback<HFModelMetadata>() {
        override fun areItemsTheSame(oldItem: HFModelMetadata, newItem: HFModelMetadata): Boolean {
            return oldItem.id == newItem.id
        }

        override fun areContentsTheSame(oldItem: HFModelMetadata, newItem: HFModelMetadata): Boolean {
            return oldItem == newItem
        }
    }
}
