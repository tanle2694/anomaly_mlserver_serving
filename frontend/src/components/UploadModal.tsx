import { ReactNode, useEffect } from "react";
import { X } from "lucide-react";

type Props = {
  onClose: () => void;
  children: ReactNode;
};

export default function UploadModal({ onClose, children }: Props) {
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close icon-button" type="button" onClick={onClose} title="Close">
          <X size={15} />
        </button>
        {children}
      </div>
    </div>
  );
}
